# SPDX-License-Identifier: Apache-2.0
# Original VTAM code. Copyright 2026 The VTAM Authors.
"""
VTAM RPC server — InferenceXArm-style.

Runs in the `genie_envisioner` (a.k.a. `vtam`) conda env. Loads the LTX
components (tokenizer / text encoder / VAE / diffusion model / scheduler /
custom pipeline) DIRECTLY via VTAM's `utils.model_utils`, mirroring the
upstream real-robot eval at `Genie-Envisioner/experiments/eval_xarm.py`
(`InferenceXArm`). It deliberately bypasses `web_infer_utils.MVActor` so the
input/output handling (state normalization, [zeros(basic_action_dim), state]
prefix, action/state split, action_type branching) is byte-for-byte identical
to the upstream pipeline.

Wire ops (same as before):

  {"op": "ping"}                              -> {"ok": True}
  {"op": "init", "args": <deploy_dict>}       -> {"ok": True, "info": {...}}
  {"op": "reset"}                              -> {"ok": True}
  {"op": "step", "obs": <obs_dict>}           -> {"ok": True, "action": np.ndarray, ...}
  {"op": "shutdown"}                           -> {"ok": True}

Wire format: 4-byte big-endian length prefix + pickle payload.

Step request payload (sent by `policy/VTAM/deploy_policy_client.py`):
    {
        "images":        np.ndarray,  # (V, H, W, 3) uint8 RGB
        "state":         np.ndarray,  # (state_dim,) float32, raw (un-normalized)
                                       # layout: [qpos(arm_dim), gripper(gripper_dim),
                                       #          tcp_xyz(3), rot6d(6)]
        "prompt":        str,
        "execution_step": int,         # number of steps consumed from the previous chunk
        "noise_seed":    int,          # optional, nonzero; diffusion noise seed, default 42
    }

Step response:
    {
        "ok":     True,
        "action": np.ndarray,         # (chunk, arm_dim + gripper_dim) float32, abs qpos targets
        "noise_seed": int,            # the seed the chunk was sampled with
    }
"""

from __future__ import annotations

import argparse
import os
import pickle
import socket
import struct
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# --- Locate VTAM (Genie-Envisioner) root ------------------------------------
# Resolve symlinks first so that running this file via the UniVTAC symlink
# (`UniVTAC/policy/VTAM/vtam_server.py` -> `VTAM/vtam_server.py`) and running
# it directly from the VTAM repo both resolve
# `_SCRIPT_DIR` to the actual VTAM repo root, regardless of how it was invoked.
_SCRIPT_DIR = Path(__file__).resolve().parent
_UNIVTAC_ROOT_GUESS = _SCRIPT_DIR.parents[1] if len(_SCRIPT_DIR.parents) >= 2 else _SCRIPT_DIR

_VTAM_ROOT_ENV = os.environ.get("VTAM_ROOT")
_VTAM_ROOT_CANDIDATES = [
    Path(_VTAM_ROOT_ENV) if _VTAM_ROOT_ENV else None,
    # Script lives directly in the VTAM repo root.
    _SCRIPT_DIR,
    # UniVTAC sibling layouts (UniVTAC user running the launcher).
    _UNIVTAC_ROOT_GUESS.parent / "VTAM",
    _UNIVTAC_ROOT_GUESS.parent / "Genie-Envisioner",
    _UNIVTAC_ROOT_GUESS / "third_party" / "VTAM",
]
VTAM_ROOT = next(
    (p for p in _VTAM_ROOT_CANDIDATES if p is not None and (p / "models").is_dir()),
    None,
)
if VTAM_ROOT is None:
    raise RuntimeError(
        "Cannot find VTAM root. Set VTAM_ROOT env var, place VTAM/Genie-Envisioner "
        "next to UniVTAC, or run vtam_server.py directly from the VTAM repo root."
    )
if str(VTAM_ROOT) not in sys.path:
    sys.path.insert(0, str(VTAM_ROOT))
print(f"[vtam_server] using VTAM_ROOT = {VTAM_ROOT}", flush=True)

import json  # noqa: E402

import torch  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402
from einops import rearrange  # noqa: E402
from yaml import Dumper as _YamlDumper, Loader as _YamlLoader, dump as _yaml_dump, load as _yaml_load  # noqa: E402

# diffusers 0.32 imports its optional TorchAO quantizer even for ordinary
# models. PyTorch <2.6 has no uint1..uint7 symbols; these aliases are only used
# while importing the unused quantizer module and never enter model execution.
for _dtype_name in ("uint1", "uint2", "uint3", "uint4", "uint5", "uint6", "uint7"):
    if not hasattr(torch, _dtype_name):
        setattr(torch, _dtype_name, torch.uint8)

from utils import import_custom_class  # noqa: E402
from utils.data_utils import get_text_conditions  # noqa: E402
from utils.model_utils import (  # noqa: E402
    load_condition_models,
    load_diffusion_model,
    load_latent_models,
    load_vae_models,
)

# xFormers is optional for inference. Some shared environments expose an
# xFormers wheel built for a different PyTorch version; diffusers detects the
# package by name and then imports it unconditionally, which can prevent the
# native PyTorch attention fallback from loading. Disable it only when its
# operators cannot be imported.
try:  # noqa: E402
    from diffusers.utils import import_utils as _diffusers_import_utils
    import xformers.ops as _xformers_ops  # noqa: F401
except Exception as _xformers_error:  # pragma: no cover - environment-specific
    _diffusers_import_utils._xformers_available = False
    print(
        f"[vtam_server] disabling incompatible xformers; using native attention: {_xformers_error}",
        flush=True,
    )

try:  # noqa: E402
    import peft.tuners.tuners_utils as _peft_tuners_utils  # noqa: F401
except Exception as _peft_error:  # pragma: no cover - environment-specific
    _diffusers_import_utils._peft_available = False
    print(
        f"[vtam_server] disabling incompatible peft; LoRA adapters are unused: {_peft_error}",
        flush=True,
    )


# Image input resolution must match VTAM training (libero_dataset / lerobot_like_dataset).
TARGET_H, TARGET_W = 192, 256


# ============== Wire helpers (same protocol as before) =====================

# Pickling a numpy array embeds the module path of numpy's internals, which
# NumPy 2 renamed from `numpy.core` to `numpy._core`. This server runs under
# NumPy 2 while the UniVTAC client is pinned to 1.26 by isaacsim, so a raw
# pickle of an ndarray fails to load on the other side with
# "No module named 'numpy._core.numeric'". Arrays therefore go over the wire
# as raw buffers plus dtype and shape, which no numpy version has an opinion
# about, and numpy scalars are demoted to plain Python numbers for the same
# reason.
_ND_TAG = "__ndarray__"


def _encode_arrays(obj):
    if isinstance(obj, np.ndarray):
        return {_ND_TAG: obj.tobytes(), "dtype": obj.dtype.str, "shape": obj.shape}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _encode_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_encode_arrays(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_encode_arrays(v) for v in obj)
    return obj


def _decode_arrays(obj):
    if isinstance(obj, dict):
        if _ND_TAG in obj:
            flat = np.frombuffer(obj[_ND_TAG], dtype=np.dtype(obj["dtype"]))
            return flat.reshape(obj["shape"]).copy()  # frombuffer is read-only
        return {k: _decode_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_arrays(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_decode_arrays(v) for v in obj)
    return obj


def _send(sock: socket.socket, payload: dict) -> None:
    blob = pickle.dumps(_encode_arrays(payload), protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(blob)) + blob)


def _recv(sock: socket.socket) -> dict:
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("client disconnected")
        header += chunk
    n = struct.unpack(">I", header)[0]
    body = bytearray()
    while len(body) < n:
        chunk = sock.recv(min(65536, n - len(body)))
        if not chunk:
            raise ConnectionError("client disconnected")
        body.extend(chunk)
    return _decode_arrays(pickle.loads(bytes(body)))


# ============== Config patching ============================================

def _patch_config_paths(
    config_file: str,
    stat_file: str | None = None,
    pretrained_model_name_or_path: str | None = None,
) -> str:
    """Mirror the previous server's config_file resolution.

    - When provided, override `pretrained_model_name_or_path` (the tokenizer /
      text encoder / VAE root), so a training config written for one machine
      can be served from another without a per-machine copy of the config.
      The override must exist: silently falling back to the HF id would load
      whatever that id currently resolves to rather than the weights trained on.
    - Otherwise replace `pretrained_model_name_or_path` with the HF id
      `Lightricks/LTX-Video` when the original path doesn't exist locally (so
      the server can load from HF cache instead of crashing on a remote path).
    - Resolve relative `data.{train,val}.stat_file` paths against `VTAM_ROOT`.
    - When provided, override both train/val statistics paths with `stat_file`.

    Returns either the original config_file path (if no patch needed) or a
    temp YAML file with the patched contents.
    """
    cfg = _yaml_load(open(config_file, "r"), Loader=_YamlLoader)
    patched = False

    if pretrained_model_name_or_path:
        override = str(Path(pretrained_model_name_or_path).expanduser())
        if not Path(override).is_dir():
            raise FileNotFoundError(
                f"pretrained_model_name_or_path override {override} is not a directory"
            )
        cfg["pretrained_model_name_or_path"] = override
        patched = True
        print(f"  [patched] pretrained_model_name_or_path -> {override}", flush=True)

    pretrained_path = cfg.get("pretrained_model_name_or_path", "")
    if pretrained_path and not Path(pretrained_path).exists():
        cfg["pretrained_model_name_or_path"] = "Lightricks/LTX-Video"
        patched = True
        print(
            "  [patched] pretrained_model_name_or_path -> Lightricks/LTX-Video",
            flush=True,
        )

    override_stat_path = None
    if stat_file:
        override_stat_path = Path(stat_file).expanduser()
        if not override_stat_path.is_absolute():
            override_stat_path = VTAM_ROOT / override_stat_path

    for split in ("train", "val"):
        if "data" in cfg and split in cfg["data"]:
            sf = cfg["data"][split].get("stat_file")
            if override_stat_path is not None:
                cfg["data"][split]["stat_file"] = str(override_stat_path)
                patched = True
            elif sf and not Path(sf).is_absolute():
                resolved = VTAM_ROOT / sf
                if resolved.exists():
                    cfg["data"][split]["stat_file"] = str(resolved)
                    patched = True

    if not patched:
        return config_file

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    _yaml_dump(cfg, tmp, Dumper=_YamlDumper)
    tmp.close()
    return tmp.name


# ============== Inference engine (mirrors InferenceXArm) ===================

class VTAMInference:
    """Direct port of `Genie-Envisioner/experiments/eval_xarm.py::InferenceXArm`.

    Public API used by the RPC server:
        - VTAMInference(args)          # `args` is the deploy_dict from client
        - reset()
        - step(images_np, state_np, prompt, execution_step) -> (chunk, qpos_dim) np.ndarray
        - meta dict with action_chunk / basic_action_dim / state_dim / arm_dim / gripper_dim
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        self.args_dict = dict(args)

        config_file = args["config_file"]
        weight_file = args["weight_file"]
        domain_name = args.get("domain_name", "univtac_pull_out_key_force_in_action")
        denoise_steps = int(args.get("denoise_steps", 10))
        n_prev = int(args.get("n_prev", 4))
        threshold = args.get("threshold", None)

        config_file = _patch_config_paths(
            config_file, args.get("stat_file"), args.get("pretrained_model_name_or_path"))
        cd = _yaml_load(open(config_file, "r"), Loader=_YamlLoader)
        ns = argparse.Namespace(**cd)
        # Keep the original behavior of overriding diffusion model weight with the
        # explicit `weight_file` from the deploy yaml.
        if isinstance(ns.diffusion_model, dict):
            ns.diffusion_model["model_path"] = weight_file
        self.ns = ns

        self.device = torch.device(args.get("device", "cuda:0"))
        self.dtype = torch.bfloat16
        self.num_inference_steps = denoise_steps
        self.n_prev = n_prev
        # `threshold` is the upstream ge-real-robot's "drop the oldest frame
        # every threshold steps" buffer parameter; preserve the same semantics.
        self.threshold = int(threshold) if threshold is not None else int(
            ns.data["train"]["action_chunk"]
        )

        # ---- stats (mirror InferenceXArm.__init__ lines 166-194) -----------
        stat_path = cd["data"]["val"]["stat_file"]
        if not Path(stat_path).is_absolute():
            stat_path = str(VTAM_ROOT / stat_path)
        with open(stat_path, "r") as f:
            stats = json.load(f) if stat_path.endswith(".json") else _yaml_load(f, Loader=_YamlLoader)

        action_space = cd["data"]["train"]["action_space"]
        resolved_action_type = str(
            args.get("action_type") or cd["data"]["train"].get("action_type", "absolute")
        )
        config_add_state = bool(cd.get("add_state", False))
        action_stat_name = f"{domain_name}_{action_space}"
        state_stat_name = f"{domain_name}_state_{action_space}"

        if action_stat_name not in stats and resolved_action_type != "relative_eef_rot6d":
            raise KeyError(
                f"Action stats key '{action_stat_name}' not found in {stat_path}. "
                f"Available keys: {list(stats.keys())}"
            )
        state_stats_available = state_stat_name in stats
        if not state_stats_available and not (
            resolved_action_type == "relative_eef_rot6d" and not config_add_state
        ):
            raise KeyError(
                f"State stats key '{state_stat_name}' not found in {stat_path}. "
                f"Available keys: {list(stats.keys())}"
            )
        if state_stats_available:
            sta_min = np.asarray(stats[state_stat_name]["q01"], dtype=np.float32)[None, :]
            sta_max = np.asarray(stats[state_stat_name]["q99"], dtype=np.float32)[None, :]
        else:
            # No-state relative models never normalize or predict state. Keep a
            # descriptive width for metadata and the output-head shape check.
            sta_min = np.zeros((1, 10), dtype=np.float32)
            sta_max = np.ones((1, 10), dtype=np.float32)
        print(
            f"[vtam_server] stats path = {stat_path}\n"
            f"[vtam_server] action stats key = {action_stat_name}\n"
            f"[vtam_server] state  stats key = {state_stat_name if state_stats_available else '(not used: no-state relative model)'}",
            flush=True,
        )

        # Use q01/q99 (min-max-style) — same as InferenceXArm.
        if action_stat_name in stats:
            act_min = np.asarray(stats[action_stat_name]["q01"], dtype=np.float32)[None, :]
            act_max = np.asarray(stats[action_stat_name]["q99"], dtype=np.float32)[None, :]
        else:
            # Relative models replace the legacy absolute action block below.
            act_min = np.empty((1, 0), dtype=np.float32)
            act_max = np.empty((1, 0), dtype=np.float32)
        self.act_min_np = act_min
        self.act_max_np = act_max
        self.states_min_np = sta_min
        self.states_max_np = sta_max

        # ---- action_type (resolved BEFORE wiring dims) ---------------------
        # Default to whatever the training config used (typically "absolute").
        action_type_arg = args.get("action_type")
        self.action_type = str(action_type_arg or ns.data["train"].get("action_type", "absolute"))
        valid_action_types = {"absolute", "relative_base", "relative_delta", "relative_eef_rot6d",
                              "relative_joint_mix"}
        if self.action_type not in valid_action_types:
            raise ValueError(
                f"action_type={self.action_type!r} not in {valid_action_types}"
            )

        # ---- chunk sizes (mirror InferenceXArm) ----------------------------
        self.add_state = bool(getattr(ns, "add_state", False))
        self.action_chunk = int(ns.data["train"]["action_chunk"])
        self.chunk_raw = int(ns.data["train"]["chunk"])

        # ---- THREE distinct model dims (do NOT assume in==out==concat) -----
        #   action_out_channels      -> diffused + predicted + buffer width
        #   model_action_in_channels -> shared action_proj_in / history-token width
        # `self.action_in_channels` keeps its historical meaning here = the
        # diffusion/output/buffer width (it was always set from action_out_channels),
        # so downstream buffer + pipeline `action_dim` usages stay correct.
        cfg = ns.diffusion_model["config"]
        self.action_out_channels = int(cfg["action_out_channels"])
        self.model_action_in_channels = int(cfg["action_in_channels"])
        self.action_in_channels = self.action_out_channels
        self.state_dim = int(sta_min.shape[1])

        if self.action_type == "relative_eef_rot6d":
            # Denorm the arm with the 13D RELATIVE stats (NOT the absolute action block).
            rel_stat_name = f"{domain_name}_relative_{action_space}"
            if rel_stat_name not in stats:
                raise KeyError(
                    f"Relative action stats key '{rel_stat_name}' not found in {stat_path}. "
                    f"Available keys: {list(stats.keys())}"
                )
            self.act_min_np = np.asarray(stats[rel_stat_name]["q01"], dtype=np.float32)[None, :]
            self.act_max_np = np.asarray(stats[rel_stat_name]["q99"], dtype=np.float32)[None, :]
            self.basic_action_dim = int(self.act_min_np.shape[1])  # 13
            # relative_eef_rot6d layout: [rel_xyz(3), rel_rot6d(6), gripper(1), force(3)].
            self.rel_pose_dim = 6       # observed base pose6 = xyz(3) + rotvec(3)
            self.rel_arm_out_dim = 9    # rel_xyz(3) + rel_rot6d(6)
            self.raw_state_dim = self.rel_pose_dim + 1  # observed state sent by client = pose6 + gripper = 7
            self.arm_dim = self.rel_pose_dim
            self.gripper_dim = 1
            self.qpos_dim = self.rel_pose_dim + self.gripper_dim  # executed = pose6 + gripper = 7 (force dropped)

            assert self.action_out_channels == self.basic_action_dim, (
                f"action_out_channels({self.action_out_channels}) != relative action dim "
                f"({self.basic_action_dim}); the 13D action-only output head is misconfigured."
            )
            if self.add_state:
                assert self.basic_action_dim + self.state_dim == self.model_action_in_channels, (
                    f"basic_action_dim({self.basic_action_dim}) + state_dim({self.state_dim}) != "
                    f"model action_in_channels({self.model_action_in_channels}); with-state pad width wrong."
                )
            else:
                assert self.model_action_in_channels == self.basic_action_dim, (
                    f"no-state model action_in_channels({self.model_action_in_channels}) must equal "
                    f"basic_action_dim({self.basic_action_dim})."
                )
        elif self.action_type == "relative_joint_mix":
            # FTP-1 "mix": [arm joint delta from the joints at inference time, gripper abs],
            # action-only head; the state input is the absolute [joints, gripper].
            rel_stat_name = f"{domain_name}_relative_{action_space}"
            if rel_stat_name not in stats:
                raise KeyError(
                    f"Relative action stats key '{rel_stat_name}' not found in {stat_path}. "
                    f"Available keys: {list(stats.keys())}"
                )
            self.act_min_np = np.asarray(stats[rel_stat_name]["q01"], dtype=np.float32)[None, :]
            self.act_max_np = np.asarray(stats[rel_stat_name]["q99"], dtype=np.float32)[None, :]
            self.basic_action_dim = int(self.act_min_np.shape[1])
            self.arm_dim = int(args.get("arm_dim", 7))
            self.gripper_dim = int(args.get("gripper_dim", 1))
            self.qpos_dim = self.arm_dim + self.gripper_dim
            self.raw_state_dim = self.state_dim
            assert self.basic_action_dim == self.qpos_dim == self.state_dim, (
                f"relative_joint_mix wants action, state and qpos all {self.qpos_dim}D; got "
                f"action {self.basic_action_dim}, state {self.state_dim}"
            )
            assert self.action_out_channels == self.basic_action_dim, (
                f"action_out_channels({self.action_out_channels}) != action dim ({self.basic_action_dim})"
            )
            assert self.add_state and self.basic_action_dim + self.state_dim == self.model_action_in_channels, (
                f"relative_joint_mix needs add_state and action_in_channels == "
                f"{self.basic_action_dim + self.state_dim}; got {self.model_action_in_channels}"
            )
        else:
            # ---------- legacy absolute / relative_base / relative_delta ----------
            # basic_action_dim / state_dim come straight from the paired stats blocks.
            self.basic_action_dim = int(act_min.shape[1])
            self.arm_dim = int(args.get("arm_dim", 7))
            self.gripper_dim = int(args.get("gripper_dim", 1))
            self.qpos_dim = self.arm_dim + self.gripper_dim
            self.raw_state_dim = self.state_dim

            assert self.basic_action_dim + self.state_dim == self.model_action_in_channels, (
                f"basic_action_dim({self.basic_action_dim}) + state_dim({self.state_dim}) "
                f"!= model action_in_channels({self.model_action_in_channels}). The action stats and "
                f"state stats are not paired with this model checkpoint."
            )
            assert self.qpos_dim <= self.state_dim, (
                f"arm_dim+gripper_dim={self.qpos_dim} but state_dim={self.state_dim}. "
                f"The qpos slice of pred_state would be out of bounds."
            )
            if not self.add_state:
                raise RuntimeError(
                    "VTAM model config has add_state=False; legacy modes require "
                    "add_state=True (the [zeros, state_norm] prefix protocol)."
                )

        # ---- gripper safety knobs (mirror RDP's deploy_abs_local.yml) ------
        # Many UniVTAC sim tasks pre-grasp the object during the curobo pre-move,
        # then the policy only needs to control the arm. Letting the model drive
        # the gripper makes it open mid-task -> object slips -> early-stop. The
        # `freeze_gripper_to_initial_width` switch (default False, opt-in via
        # deploy yaml) clamps the gripper output of every step to whatever
        # gripper width the sim measured on the first observation, identical to
        # RDP's policy/RDP/rdp_server.py:_get_target_gripper_width semantics.
        self.freeze_gripper_to_initial_width = bool(
            args.get("freeze_gripper_to_initial_width", False)
        )
        self.print_gripper_debug = bool(args.get("print_gripper_debug", False))
        # Optional physical clip of gripper width; defaults are action_type-aware so
        # a deploy yaml need not override them per task:
        #   relative_eef_rot6d (hanoi): raw left_hand_joints observed range [0.031, 0.084]
        #     (bimodal open~0.031 / closed~0.084). Clip to [0, 0.085] so the "closed"
        #     width (0.084) is never truncated (the old [0, 0.039] default would).
        #   legacy (univtac): [0, 0.039] m matches the univtac stats observed range.
        if self.action_type == "relative_eef_rot6d":
            _grip_min_default, _grip_max_default = 0.0, 0.085
        else:
            _grip_min_default, _grip_max_default = 0.0, 0.039
        self.gripper_min = float(args.get("gripper_min", _grip_min_default))
        self.gripper_max = float(args.get("gripper_max", _grip_max_default))
        # Index of the gripper column inside the qpos action (right after the
        # arm joints, before any extra force / pose dims).
        self.gripper_action_index = int(args.get("gripper_action_index", self.arm_dim))
        self.initial_gripper_width: Optional[float] = None
        # relative_eef_rot6d: observed base pose6 latched each step for compose-back.
        self._current_base6: Optional[np.ndarray] = None
        # relative_joint_mix: observed arm joints latched each step, the chunk's base.
        self._current_base_q: Optional[np.ndarray] = None

        # ---- image transforms (identical to libero_dataset.py / InferenceXArm) -
        self.pixel_transforms_resize = transforms.Compose([
            transforms.Resize((TARGET_H, TARGET_W)),
        ])
        self.pixel_transforms_norm = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False),
        ])

        # ---- buffers (mirror InferenceXArm) --------------------------------
        self.obs_buf: list[torch.Tensor] = []
        self.frame_buf: list[torch.Tensor] = []
        self.count = 0
        self.action_buffer = torch.zeros(
            self.action_chunk, self.action_in_channels, dtype=self.dtype
        )

        print(
            f"[vtam_server] dims: action_in_channels={self.action_in_channels}, "
            f"state_dim={self.state_dim}, basic_action_dim={self.basic_action_dim}, "
            f"qpos_dim={self.qpos_dim}, action_chunk={self.action_chunk}, "
            f"action_type={self.action_type}, threshold={self.threshold}",
            flush=True,
        )

        self._prepare_models()
        # Compute VAE-based chunk after VAE is loaded.
        self.chunk = (self.chunk_raw - 1) // self.TEMPORAL_DOWN_RATIO + 1

    # ---- model loading (mirror InferenceXArm.prepare_models) ---------------
    def _prepare_models(self) -> None:
        ns = self.ns
        device = self.device
        dtype = self.dtype

        tokenizer_class = import_custom_class(
            ns.tokenizer_class, getattr(ns, "tokenizer_class_path", "transformers")
        )
        textenc_class = import_custom_class(
            ns.textenc_class, getattr(ns, "textenc_class_path", "transformers")
        )
        cond_models = load_condition_models(
            tokenizer_class,
            textenc_class,
            ns.pretrained_model_name_or_path
            if not hasattr(ns, "tokenizer_pretrained_model_name_or_path")
            else ns.tokenizer_pretrained_model_name_or_path,
            load_weights=ns.load_weights,
        )
        self.tokenizer = cond_models["tokenizer"]
        self.text_encoder = cond_models["text_encoder"].to(device, dtype=dtype).eval()

        text_uncond = get_text_conditions(self.tokenizer, self.text_encoder, prompt="")
        self.uncond_prompt_embeds = text_uncond["prompt_embeds"]
        self.uncond_prompt_attention_mask = text_uncond["prompt_attention_mask"]

        vae_class = import_custom_class(
            ns.vae_class, getattr(ns, "vae_class_path", "transformers")
        )
        if getattr(ns, "vae_path", False):
            self.vae = load_vae_models(vae_class, ns.vae_path).to(device, dtype=dtype).eval()
        else:
            self.vae = (
                load_latent_models(vae_class, ns.pretrained_model_name_or_path)["vae"]
                .to(device, dtype=dtype)
                .eval()
            )
        if self.vae is not None:
            if getattr(ns, "enable_slicing", False):
                self.vae.enable_slicing()
            if getattr(ns, "enable_tiling", False):
                self.vae.enable_tiling()
        self.SPATIAL_DOWN_RATIO = getattr(self.vae, "spatial_compression_ratio", 1)
        self.TEMPORAL_DOWN_RATIO = getattr(self.vae, "temporal_compression_ratio", 1)

        diffusion_model_class = import_custom_class(
            ns.diffusion_model_class,
            getattr(ns, "diffusion_model_class_path", "transformers"),
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=ns.diffusion_model["model_path"],
            load_weights=ns.load_weights and getattr(ns, "load_diffusion_model_weights", True),
            **ns.diffusion_model["config"],
        ).to(device, dtype=dtype)

        diffusion_scheduler_class = import_custom_class(
            ns.diffusion_scheduler_class,
            getattr(ns, "diffusion_scheduler_class_path", "diffusers"),
        )
        if hasattr(ns, "diffusion_scheduler_args"):
            self.scheduler = diffusion_scheduler_class(**ns.diffusion_scheduler_args)
        else:
            self.scheduler = diffusion_scheduler_class()

        self.pipeline_class = import_custom_class(
            ns.pipeline_class, getattr(ns, "pipeline_class_path", "diffusers")
        )
        self.pipeline = self.pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer, self.diffusion_model
        )

    # ---- per-episode state -------------------------------------------------
    def reset(self) -> None:
        self.obs_buf = []
        self.frame_buf = []
        self.count = 0
        self.action_buffer = torch.zeros(
            self.action_chunk, self.action_in_channels, dtype=self.dtype
        )
        # Forget the held gripper width so the next episode picks up a fresh
        # initial value from the first observation (mirrors RDP rdp_server.py).
        self.initial_gripper_width = None
        self._current_base6 = None
        self._current_base_q = None

    # ---- one inference step ------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        images_np: np.ndarray,
        state_np: np.ndarray,
        prompt: str,
        execution_step: int,
        noise_seed: int = 42,
    ) -> np.ndarray:
        """Run one inference cycle.

        Args:
            images_np: (V, H, W, 3) uint8 RGB; resized + normalized inside.
            state_np: (state_dim,) float32 raw state matching the layout of
                state stats: [qpos, gripper, tcp_xyz, rot6d].
            prompt: language instruction.
            execution_step: number of steps the client consumed since the last
                inference; controls the frame-buffer rolling threshold.

        Returns:
            (action_chunk, qpos_dim) float32 numpy array — absolute joint+gripper
            targets ready to feed `task.take_action(..., action_type="qpos")`.
        """
        if not isinstance(images_np, np.ndarray):
            images_np = np.asarray(images_np)
        if images_np.ndim != 4 or images_np.shape[-1] != 3:
            raise ValueError(
                f"images_np must be (V, H, W, 3); got {images_np.shape}"
            )

        # ---- image preprocessing (mirror InferenceXArm.play 297-306) ------
        if images_np.dtype == np.uint8:
            obs = torch.from_numpy(images_np.copy()).permute(0, 3, 1, 2).float() / 255.0
            obs = self.pixel_transforms_resize(obs)
            obs = self.pixel_transforms_norm(obs)
        else:
            obs = torch.from_numpy(images_np)

        v, c, h, w = obs.shape
        obs = obs.to(self.device, dtype=self.dtype)

        # ---- state preprocessing (mirror eval_xarm.py 632-638) ------------
        state_np = np.asarray(state_np, dtype=np.float32).reshape(-1)
        if state_np.shape[0] != self.raw_state_dim:
            raise ValueError(
                f"state has {state_np.shape[0]} dims but the server expects raw_state_dim="
                f"{self.raw_state_dim} for action_type={self.action_type}. "
                f"(relative_eef_rot6d expects the raw 7D [pose6, gripper]; legacy expects "
                f"the full model state layout [qpos({self.arm_dim}), gripper, tcp_xyz(3), rot6d(6)].)"
            )
        # Latch initial gripper width from the very first observation each
        # episode; reset() clears this so a new seed starts fresh. Gripper sits
        # right after the arm/pose block in the RAW observed state.
        if self.initial_gripper_width is None:
            self.initial_gripper_width = float(state_np[self.arm_dim])

        if self.action_type == "relative_eef_rot6d":
            # Observed base pose6 (xyz + rotvec); used to compose the absolute target below.
            self._current_base6 = state_np[: self.rel_pose_dim].astype(np.float32).copy()  # (6,)
            # Convert raw 7D observed state -> 10D rot6d state: [xyz(3), abs_rot6d(6), gripper(1)].
            from data.utils.pose_math import abs_pose6_to_pose9
            arm9 = abs_pose6_to_pose9(state_np[None, : self.rel_pose_dim])                 # (1,9)
            current_state = np.concatenate(
                [arm9, state_np[None, self.rel_pose_dim:]], axis=1
            ).astype(np.float32)                                                          # (1,10)
        else:
            current_state = state_np[None, :]  # (1, state_dim)
            if self.action_type == "relative_joint_mix":
                self._current_base_q = state_np[: self.arm_dim].astype(np.float32).copy()

        if self.add_state:
            state_norm = (current_state - self.states_min_np) / (
                self.states_max_np - self.states_min_np + 1e-6
            )
            state_norm = state_norm * 2.0 - 1.0
            action_zeros = np.zeros((1, self.basic_action_dim), dtype=np.float32)
            # state_in width = basic_action_dim + state_dim = model_action_in_channels.
            state_in = np.concatenate([action_zeros, state_norm], axis=1).astype(np.float32)

            history_action_state = (
                torch.from_numpy(state_in).to(self.device, dtype=self.dtype)
            )
            # pipeline.infer expects shape (B, T, C); add T axis.
            if history_action_state.dim() == 2:
                history_action_state = history_action_state.unsqueeze(1)
            assert (
                history_action_state.dim() == 3
                and history_action_state.shape[-1] == self.model_action_in_channels
            ), (
                f"history_action_state shape {tuple(history_action_state.shape)} "
                f"does not match (B, T, model_action_in_channels={self.model_action_in_channels})"
            )
        else:
            # no-state variant: no conditioning token; the pipeline's forward pad is a no-op.
            history_action_state = None

        # ---- frame buffer rolling (mirror InferenceXArm.play 322-338) -----
        if not self.obs_buf:
            self.obs_buf = [obs] * self.n_prev
            self.count = self.threshold - 1
            self.frame_buf = [self.obs_buf[-1]]
        else:
            if execution_step == 0:
                # Nothing consumed; reuse previous buffer chunk verbatim.
                return self._action_buffer_to_qpos_np()
            self.count += execution_step
            if self.count >= self.threshold:
                self.count = 0
                self.obs_buf.pop(0)
                self.obs_buf[-1] = self.frame_buf[0]
                self.obs_buf.append(obs)
            else:
                self.obs_buf[-1] = obs
            self.frame_buf = [self.obs_buf[-1]]

        # ---- pipeline.infer call (mirror InferenceXArm.play 340-366) ------
        obs_tensor = torch.stack(self.obs_buf, dim=1)  # (V, T, C, H, W)
        obs_tensor = rearrange(obs_tensor, "v t c h w -> c v t h w")
        obs_tensor = obs_tensor.unsqueeze(0)  # (1, C, V, T, H, W)
        obs_tensor = rearrange(obs_tensor, "b c v t h w -> (b v) c t h w")

        pred_all = self.pipeline.infer(
            image=obs_tensor,
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=self.num_inference_steps,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            guidance_scale=1.0,
            height=h,
            width=w,
            n_view=v,
            return_action=True,
            return_video=False,
            chunk=self.chunk,
            action_chunk=self.action_chunk,
            history_action_state=history_action_state,
            noise_seed=noise_seed,
            pixel_wise_timestep=getattr(self.ns, "pixel_wise_timestep", True),
            n_chunk=1,
            n_prev=self.n_prev,
            # diffusion SAMPLE width = action_out_channels (13); the pipeline zero-pads
            # the sample to model_action_in_channels (23) for the forward pass only.
            action_dim=self.action_out_channels,
        )[0]

        actions_pred_full = pred_all["action"].detach()[0]  # (chunk, action_out_channels=13)
        self.action_buffer = actions_pred_full.clone()

        # ---- split + denorm + action_type branch (mirror eval_xarm.py 654-673) -
        actions_norm_full = actions_pred_full.detach().cpu().numpy().astype(np.float32)
        bad = self.basic_action_dim
        pred_action_norm = actions_norm_full[:, :bad]
        pred_state_norm = actions_norm_full[:, bad:]

        pred_action_abs = (pred_action_norm + 1.0) / 2.0 * (
            self.act_max_np - self.act_min_np + 1e-6
        ) + self.act_min_np
        # Legacy modes concatenate state into the output head (pred_state_norm width == state_dim).
        # relative_eef_rot6d has a TRUE action-only head, so pred_state_norm is empty -> skip.
        if pred_state_norm.shape[1] == self.state_dim:
            pred_state_abs = (pred_state_norm + 1.0) / 2.0 * (
                self.states_max_np - self.states_min_np + 1e-6
            ) + self.states_min_np
        else:
            pred_state_abs = None

        if self.action_type == "relative_eef_rot6d":
            # pred_action_abs (RAW) layout = [rel_xyz(3), rel_rot6d(6), gripper(1), force(3)].
            # The model predicts a FUTURE EEF TRAJECTORY TARGET (relative to the current
            # observed pose), NOT a raw controller command -- so we compose it back onto the
            # current base pose to get the absolute pose to execute.
            # Compose the 9D relative arm with the observed base pose6 -> absolute EEF pose6.
            # Execute pose6 + gripper only; force is a training-only regularizer and is dropped.
            from data.utils.pose_math import compose_rot6d_to_pose6
            rel9 = pred_action_abs[:, : self.rel_arm_out_dim]                                   # (chunk, 9)
            gripper_col = pred_action_abs[:, self.rel_arm_out_dim : self.rel_arm_out_dim + 1]   # (chunk, 1)
            abs_pose6 = compose_rot6d_to_pose6(self._current_base6, rel9)                       # (chunk, 6)
            actions = np.concatenate([abs_pose6, gripper_col], axis=1).astype(np.float32, copy=False)  # (chunk, 7)
        elif self.action_type == "relative_joint_mix":
            # Every row is relative to the same base (the joints at this inference), not
            # cumulative; the gripper column is already absolute.
            actions = pred_action_abs[:, : self.qpos_dim].astype(np.float32, copy=True)
            actions[:, : self.arm_dim] += self._current_base_q[None, :]
        elif self.action_type == "absolute":
            # Upstream uses pred_state (NOT pred_action) for absolute mode.
            # Slice [:qpos_dim] = (arm_dim + gripper_dim) so the rest of the
            # state prediction (tcp_xyz, rot6d) is harmlessly dropped.
            actions = pred_state_abs[:, : self.qpos_dim].astype(np.float32, copy=False)
        elif self.action_type == "relative_base":
            # action_norm is offset from current state; share base across chunk.
            arm = self.arm_dim
            action_norm = pred_action_norm.copy()
            action_norm[:, :arm] = action_norm[:, :arm] + state_norm[:, :arm]
            relbase_abs = (action_norm + 1.0) / 2.0 * (
                self.act_max_np - self.act_min_np + 1e-6
            ) + self.act_min_np
            actions = relbase_abs[:, : self.qpos_dim].astype(np.float32, copy=False)
        else:  # relative_delta
            arm = self.arm_dim
            integrated_norm = state_norm.copy()  # (1, state_dim)
            actions = np.zeros(
                (pred_action_norm.shape[0], self.basic_action_dim), dtype=np.float32
            )
            for k in range(pred_action_norm.shape[0]):
                integrated_norm[:, :arm] = (
                    integrated_norm[:, :arm] + pred_action_norm[k : k + 1, :arm]
                )
                # Non-arm dims (gripper + force) snap to predicted action directly.
                integrated_norm[:, arm:bad] = pred_action_norm[k : k + 1, arm:bad]
                row = (integrated_norm + 1.0) / 2.0 * (
                    self.act_max_np[:, :bad] - self.act_min_np[:, :bad] + 1e-6
                ) + self.act_min_np[:, :bad]
                actions[k] = row[0]
            actions = actions[:, : self.qpos_dim].astype(np.float32, copy=False)

        # ---- gripper safety override (mirror RDP rdp_server.py) ------------
        # Apply per-step gripper handling AFTER action_type branching so all
        # three modes share the same logic. The gripper column lives at
        # `gripper_action_index` (default = arm_dim).
        gi = self.gripper_action_index
        if 0 <= gi < actions.shape[1]:
            raw_grip = actions[:, gi].copy()
            if self.freeze_gripper_to_initial_width:
                if self.initial_gripper_width is None:
                    raise RuntimeError(
                        "freeze_gripper_to_initial_width=True but no initial "
                        "gripper observed yet (call reset() then step()).")
                actions[:, gi] = float(self.initial_gripper_width)
            actions[:, gi] = np.clip(
                actions[:, gi], self.gripper_min, self.gripper_max
            )
            if self.print_gripper_debug:
                obs_grip = (
                    float(self.initial_gripper_width)
                    if self.initial_gripper_width is not None
                    else float("nan")
                )
                mode = (
                    "frozen" if self.freeze_gripper_to_initial_width else "policy"
                )
                # Print only the first row of the chunk to keep logs sane.
                print(
                    f"[VTAM gripper] mode={mode} initial={obs_grip:.6f} "
                    f"pred[0]={raw_grip[0]:.6f} applied[0]={actions[0, gi]:.6f}",
                    flush=True,
                )

        return actions

    def _action_buffer_to_qpos_np(self) -> np.ndarray:
        """Reuse the previously cached chunk when the client requests no new
        inference (execution_step==0). Mirrors `MVActor.play` early-return."""
        cached = self.action_buffer.detach().cpu().numpy().astype(np.float32)
        # The cached buffer is in the model's normalized (28D) space; we don't
        # have enough state context here to denormalize properly without re-
        # running the action_type branch. The simplest correct behavior is to
        # not cache (clients should not send execution_step==0 in this server),
        # but if they do we return zeros so the assertion fires loudly.
        return np.zeros(
            (cached.shape[0], self.qpos_dim), dtype=np.float32
        )


# ============== Server =====================================================

class VTAMServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 5006) -> None:
        self.host = host
        self.port = port
        self.engine: Optional[VTAMInference] = None

    def _handle_init(self, args: dict) -> dict:
        print("[vtam_server] Loading model...", flush=True)
        for k in ("config_file", "weight_file", "domain_name", "denoise_steps", "n_prev", "threshold"):
            if k in args:
                print(f"  {k:14s} = {args[k]}", flush=True)
        self.engine = VTAMInference(args)
        print("[vtam_server] Model loaded successfully.", flush=True)
        return {
            "ok": True,
            "info": {
                "action_chunk": self.engine.action_chunk,
                "action_dim": self.engine.qpos_dim,
                "basic_action_dim": self.engine.basic_action_dim,
                "state_dim": self.engine.state_dim,
                "action_type": self.engine.action_type,
                "action_space": self.engine.ns.data["train"].get("action_space", "joint"),
            },
        }

    def _handle_step(self, obs: dict) -> dict:
        if self.engine is None:
            return {"ok": False, "error": "Engine not initialized. Send 'init' first."}
        # The pipeline treats a falsy seed as "unseeded" and also switches the VAE
        # encode from argmax to sampling, so 0 is rejected rather than passed on.
        noise_seed = int(obs.get("noise_seed", 42))
        if noise_seed == 0:
            return {"ok": False, "error": "noise_seed must be nonzero"}
        action = self.engine.step(
            images_np=obs["images"],
            state_np=obs["state"],
            prompt=obs.get("prompt", ""),
            execution_step=int(obs.get("execution_step", 1)),
            noise_seed=noise_seed,
        )
        return {"ok": True, "action": action, "noise_seed": noise_seed}

    def _handle_reset(self) -> dict:
        if self.engine is not None:
            self.engine.reset()
        return {"ok": True}

    def _handle_request(self, request: dict) -> dict:
        op = request.get("op")
        if op == "ping":
            return {"ok": True}
        if op == "init":
            return self._handle_init(request["args"])
        if op == "step":
            return self._handle_step(request["obs"])
        if op == "reset":
            return self._handle_reset()
        if op == "shutdown":
            return {"ok": True}
        return {"ok": False, "error": f"Unknown op: {op}"}

    def serve(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        print(f"[vtam_server] Listening on {self.host}:{self.port}", flush=True)

        while True:
            conn, addr = srv.accept()
            print(f"[vtam_server] Connection from {addr}", flush=True)
            try:
                while True:
                    request = _recv(conn)
                    try:
                        response = self._handle_request(request)
                    except Exception:
                        response = {
                            "ok": False,
                            "error": "Server exception",
                            "traceback": traceback.format_exc(),
                        }
                    _send(conn, response)
                    if request.get("op") == "shutdown":
                        print("[vtam_server] Shutdown requested.", flush=True)
                        conn.close()
                        return
            except (ConnectionError, OSError) as e:
                print(f"[vtam_server] Client {addr} disconnected: {e}", flush=True)
            except Exception:
                print(f"[vtam_server] Error:\n{traceback.format_exc()}", flush=True)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="VTAM RPC Server (InferenceXArm-style)")
    parser.add_argument("--host", default=os.environ.get("VTAM_SERVER_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("VTAM_SERVER_PORT", "5006"))
    )
    args = parser.parse_args()

    server = VTAMServer(host=args.host, port=args.port)
    server.serve()


if __name__ == "__main__":
    main()
