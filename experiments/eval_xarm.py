# SPDX-License-Identifier: Apache-2.0
# Original VTAM code (real-robot evaluation on xArm6). Copyright 2026 The VTAM Authors.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XArm Diffusion Policy Deployment Script (Manual-Trigger Mode)

Flow per cycle:
  1) Press Enter to trigger
  2) Capture images from cameras
  3) Read robot state
  4) Run model inference
  5) Execute predicted actions at 30Hz (direct send, no smoothing, identical to replay)
  6) Wait for next Enter

Image preprocessing uses the exact same torchvision transforms as training.
"""

import os
import sys
import argparse
import time
import csv

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from yaml import load, Loader
from einops import rearrange
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.model_utils import (
    load_condition_models,
    load_latent_models,
    load_vae_models,
    load_diffusion_model,
)
from utils import import_custom_class
from utils.data_utils import get_text_conditions


TARGET_H, TARGET_W = 192, 256


def euler_to_rotation_6d(rx: float, ry: float, rz: float) -> np.ndarray:
    """Euler angles (xyz convention, radians) -> 6D continuous rotation."""
    R = Rotation.from_euler("xyz", [rx, ry, rz]).as_matrix()  # 3x3
    return R[:, :2].T.flatten().astype(np.float32)  # (6,)
INITIAL_GRIPPER_POS = 800
FIXED_GRIPPER_POS = None  # Force gripper to this width (set None to use model prediction)

# ---------------------------------------------------------------------------
# GelSight optical-flow force estimation (real-time)
# ---------------------------------------------------------------------------

FORCE_BASELINE_N = 10  # first N frames averaged as no-contact reference


def _extract_marker_mask(gray: np.ndarray, thresh: int = 70) -> np.ndarray:
    """Extract dark-marker binary mask from GelSight grayscale image."""
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _compute_flow_forces(
    ref_gray: np.ndarray, cur_gray: np.ndarray, marker_mask: np.ndarray | None = None
) -> tuple[float, float, float]:
    """Return (fx, fy, fz) from Farneback optical flow between ref and cur."""
    ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 1.0)
    cur_blur = cv2.GaussianBlur(cur_gray, (5, 5), 1.0)
    flow = cv2.calcOpticalFlowFarneback(
        ref_blur, cur_blur, None,
        pyr_scale=0.5, levels=5, winsize=21,
        iterations=5, poly_n=7, poly_sigma=1.5, flags=0,
    )
    mask_bool = marker_mask > 0 if marker_mask is not None else np.ones(ref_gray.shape, dtype=bool)
    dx, dy = flow[:, :, 0], flow[:, :, 1]
    fx = float(np.mean(dx[mask_bool]))
    fy = float(np.mean(dy[mask_bool]))
    # fz ≈ divergence of smoothed flow field
    dx_s = cv2.GaussianBlur(dx, (7, 7), 2.0)
    dy_s = cv2.GaussianBlur(dy, (7, 7), 2.0)
    fz = float(np.mean((np.gradient(dx_s, axis=1) + np.gradient(dy_s, axis=0))[mask_bool]))
    return fx, fy, fz


class GelSightForceEstimator:
    """Maintains a no-contact reference and computes real-time force from new frames."""

    def __init__(self, baseline_n: int = FORCE_BASELINE_N):
        self.baseline_n = baseline_n
        self.ref_gray: np.ndarray | None = None
        self.marker_mask: np.ndarray | None = None
        self._baseline_accum: list[np.ndarray] = []
        self._ready = False

    def _to_gray(self, img_bgr: np.ndarray) -> np.ndarray:
        """Convert BGR image to grayscale, resize to a fixed size for consistency."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240))
        return gray

    def update(self, img_bgr: np.ndarray) -> tuple[float, float, float]:
        """Feed a new GelSight BGR frame, return (fx, fy, fz).

        During the first `baseline_n` calls the frame is accumulated
        as the no-contact reference and (0, 0, 0) is returned.
        """
        gray = self._to_gray(img_bgr)
        if not self._ready:
            self._baseline_accum.append(gray.astype(np.float32))
            if len(self._baseline_accum) >= self.baseline_n:
                self.ref_gray = np.mean(self._baseline_accum, axis=0).astype(np.uint8)
                self.marker_mask = _extract_marker_mask(self.ref_gray)
                self._ready = True
                print(f"   🔬 GelSight force baseline captured ({self.baseline_n} frames)")
            return 0.0, 0.0, 0.0
        return _compute_flow_forces(self.ref_gray, gray, self.marker_mask)


def process_image(img):
    """Resize to training size (256x192) without crop."""
    return cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi



class InferenceXArm:
    def __init__(
        self,
        config_file: str,
        model_path: str | None,
        device: str,
        exec_step: int,
        threshold: int,
        num_inference_steps: int,
    ) -> None:
        cd = load(open(config_file, "r"), Loader=Loader)
        args = argparse.Namespace(**cd)
        self.args = args

        if model_path is not None:
            self.args.diffusion_model["model_path"] = model_path

        self.device = device
        self.dtype = torch.bfloat16
        self.exec_step = exec_step
        self.threshold = threshold
        self.num_inference_steps = num_inference_steps

        self.action_dim = cd["diffusion_model"]["config"].get(
            "action_out_channels", cd["diffusion_model"]["config"]["action_in_channels"]
        )
        self.add_state = cd.get("add_state", False)
        self.n_prev = cd["data"]["train"]["n_previous"]
        self.chunk_raw = cd["data"]["train"]["chunk"]
        self.action_chunk = cd["data"]["train"]["action_chunk"]

        # stats (kept both numpy and torch forms)
        stat_path = cd["data"]["val"]["stat_file"]
        with open(stat_path, "r") as f:
            stats = (
                load(f, Loader=Loader)
                if stat_path.endswith(".yaml")
                else __import__("json").load(f)
            )

        domain = cd["data"]["train"]["domains"][0]
        action_space = cd["data"]["train"]["action_space"]
        action_stat_name = f"{domain}_{action_space}"
        state_stat_name = f"{domain}_state_{action_space}"

        act_min_np = np.array(stats[action_stat_name]["q01"], dtype=np.float32)[None, :]
        act_max_np = np.array(stats[action_stat_name]["q99"], dtype=np.float32)[None, :]
        states_min_np = np.array(stats[state_stat_name]["q01"], dtype=np.float32)[None, :]
        states_max_np = np.array(stats[state_stat_name]["q99"], dtype=np.float32)[None, :]

        self.act_min_np = act_min_np
        self.act_max_np = act_max_np
        self.states_min_np = states_min_np
        self.states_max_np = states_max_np

        # basic_action_dim: read directly from action stats
        # e.g. 7 (chip_reguralized: joint6+gripper1), 10 (force_in_action: joint6+gripper1+force3)
        self.basic_action_dim = act_min_np.shape[1]
        state_dim = states_min_np.shape[1]
        print(f"   action_dim={self.action_dim}, state_dim={state_dim}, basic_action_dim={self.basic_action_dim}")

        # torch (move to device once)
        self.act_min = torch.tensor(act_min_np, device=self.device, dtype=self.dtype)
        self.act_max = torch.tensor(act_max_np, device=self.device, dtype=self.dtype)
        self.states_min = torch.tensor(states_min_np, device=self.device, dtype=self.dtype)
        self.states_max = torch.tensor(states_max_np, device=self.device, dtype=self.dtype)

        # Image transforms – identical to training (libero_dataset.py / lerobot_like_dataset.py)
        self.pixel_transforms_resize = transforms.Compose([
            transforms.Resize((TARGET_H, TARGET_W)),
        ])
        self.pixel_transforms_norm = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.obs = []
        self.buffer = []
        self.count = 0
        self.action_buffer = torch.zeros(self.action_chunk, self.action_dim, dtype=self.dtype)

        self.prepare_models()

    def prepare_models(self):
        tokenizer_class = import_custom_class(
            self.args.tokenizer_class,
            getattr(self.args, "tokenizer_class_path", "transformers"),
        )
        textenc_class = import_custom_class(
            self.args.textenc_class,
            getattr(self.args, "textenc_class_path", "transformers"),
        )
        cond_models = load_condition_models(
            tokenizer_class,
            textenc_class,
            self.args.pretrained_model_name_or_path
            if not hasattr(self.args, "tokenizer_pretrained_model_name_or_path")
            else self.args.tokenizer_pretrained_model_name_or_path,
            load_weights=self.args.load_weights,
        )
        self.tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
        self.text_encoder = text_encoder.to(self.device, dtype=self.dtype).eval()

        self.text_uncond = get_text_conditions(self.tokenizer, self.text_encoder, prompt="")
        self.uncond_prompt_embeds = self.text_uncond["prompt_embeds"]
        self.uncond_prompt_attention_mask = self.text_uncond["prompt_attention_mask"]

        vae_class = import_custom_class(
            self.args.vae_class, getattr(self.args, "vae_class_path", "transformers")
        )
        if getattr(self.args, "vae_path", False):
            self.vae = load_vae_models(vae_class, self.args.vae_path).to(self.device, dtype=self.dtype).eval()
        else:
            self.vae = load_latent_models(vae_class, self.args.pretrained_model_name_or_path)["vae"].to(
                self.device, dtype=self.dtype
            ).eval()

        if self.vae is not None:
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()
        self.SPATIAL_DOWN_RATIO = getattr(self.vae, "spatial_compression_ratio", 1)
        self.TEMPORAL_DOWN_RATIO = getattr(self.vae, "temporal_compression_ratio", 1)
        self.chunk = (self.chunk_raw - 1) // self.TEMPORAL_DOWN_RATIO + 1

        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class,
            getattr(self.args, "diffusion_model_class_path", "transformers"),
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.args.diffusion_model["model_path"],
            load_weights=self.args.load_weights and getattr(self.args, "load_diffusion_model_weights", True),
            **self.args.diffusion_model["config"],
        ).to(self.device, dtype=self.dtype)

        diffusion_scheduler_class = import_custom_class(
            self.args.diffusion_scheduler_class,
            getattr(self.args, "diffusion_scheduler_class_path", "diffusers"),
        )
        if hasattr(self.args, "diffusion_scheduler_args"):
            self.scheduler = diffusion_scheduler_class(**self.args.diffusion_scheduler_args)
        else:
            self.scheduler = diffusion_scheduler_class()

        self.pipeline_class = import_custom_class(
            self.args.pipeline_class, getattr(self.args, "pipeline_class_path", "diffusers")
        )
        self.pipeline = self.pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer, self.diffusion_model
        )

    @torch.no_grad()
    def play(self, obs, prompt, execution_step=1, state=None, return_normalized=False, return_full=False):
        """
        obs: (V,H,W,3) uint8 RGB or (V,3,H,W) float in [-1,1]
        state: numpy or torch; expected already normalized to [-1,1] if add_state is enabled
        """
        # Normalize obs using the exact same pipeline as training:
        #   1) uint8 → torch float [0,1]
        #   2) torchvision.transforms.Resize  (bilinear, same as training)
        #   3) torchvision.transforms.Normalize(0.5, 0.5, 0.5) → [-1,1]
        if isinstance(obs, np.ndarray):
            if obs.dtype == np.uint8:
                obs = torch.from_numpy(obs.copy()).permute(0, 3, 1, 2).float() / 255.0  # (V,C,H,W) [0,1]
                obs = self.pixel_transforms_resize(obs)   # Resize to (TARGET_H, TARGET_W)
                obs = self.pixel_transforms_norm(obs)      # Normalize to [-1,1]
            else:
                obs = torch.from_numpy(obs)

        v, c, h, w = obs.shape
        obs = obs.to(self.device, dtype=self.dtype)  # do NOT call obs.max().cpu() (sync)

        history_action_state = state
        if self.add_state and history_action_state is not None:
            if isinstance(history_action_state, np.ndarray):
                history_action_state = torch.from_numpy(history_action_state)
            history_action_state = history_action_state.to(self.device, dtype=self.dtype)
            while len(history_action_state.shape) < 3:
                history_action_state = history_action_state.unsqueeze(dim=0)
            # keep original assertion, but now your state should match expected dim
            assert history_action_state.shape[-1] == self.action_dim, (
                f"history_action_state last dim {history_action_state.shape[-1]} != action_dim {self.action_dim}"
            )
        else:
            history_action_state = None

        # frame buffering logic
        if not self.obs:
            self.obs = [obs] * self.n_prev
            self.count = self.threshold - 1
            self.buffer = [self.obs[-1]]
        else:
            if execution_step == 0:
                return self.action_buffer
            self.count += execution_step
            if self.count >= self.threshold:
                self.count = 0
                self.obs.pop(0)
                self.obs[-1] = self.buffer[0]
                self.obs.append(obs)
            else:
                self.obs[-1] = obs
            self.buffer = [self.obs[-1]]

        obs_tensor = torch.stack(self.obs, dim=1)  # V,T,C,H,W
        obs_tensor = rearrange(obs_tensor, "v t c h w -> c v t h w")
        obs_tensor = obs_tensor.unsqueeze(0)
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
            history_action_state=history_action_state if self.add_state else None,
            noise_seed=42,
            pixel_wise_timestep=self.args.pixel_wise_timestep,
            n_chunk=1,
            n_prev=self.n_prev,
            action_dim=self.action_dim,
        )[0]

        actions_pred_full = pred_all["action"].detach()[0]  # (action_chunk, action_dim) on device
        # Use the first basic_action_dim dims as action output (7 for normal, 10 for force_in_action).
        actions_pred = actions_pred_full[:, :self.basic_action_dim]

        if return_normalized:
            self.action_buffer = actions_pred.clone()
            if return_full:
                return actions_pred.detach().cpu(), actions_pred_full.detach().cpu()
            return actions_pred.detach().cpu()  # normalized (for relative modes)

        # denorm to real action space (action stats)
        actions_pred = (actions_pred + 1) / 2
        actions_pred = actions_pred * (self.act_max - self.act_min + 1e-6) + self.act_min

        self.action_buffer = actions_pred.clone()
        if return_full:
            return actions_pred.detach().cpu(), actions_pred_full.detach().cpu()
        return actions_pred.detach().cpu()  # return on CPU for control thread


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="wipe the plate")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--exec_step", type=int, default=30)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=20)
    parser.add_argument("--ip", type=str, required=True, help="xArm controller IP address")

    parser.add_argument("--base_cam_index", type=int, default=4)
    parser.add_argument("--third_cam_index", type=int, default=10)
    parser.add_argument("--gelsight_cam_index", type=int, default=12)
    parser.add_argument("--use_third", action="store_true", help="Use third view camera")
    parser.add_argument("--use_gelsight", action="store_true", help="Use gelsight camera")
    parser.add_argument("--gelsight_force_only", action="store_true",
                        help="Open gelsight camera for force extraction only; do NOT feed its image to the model as a view")

    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--smoothing", type=float, default=0.15)
    parser.add_argument("--max_step", type=float, default=0.10, help="Max joint delta per 30Hz step (rad/step)")
    parser.add_argument("--debug_every", type=int, default=10, help="Write debug image every N cycles (0 disables)")
    parser.add_argument("--debug_action_every", type=int, default=0, help="Print action debug every N cycles (0 disables)")
    parser.add_argument("--log_dir", type=str, default="eval_logs", help="Directory for 1Hz/30Hz CSV logs")
    parser.add_argument(
        "--action_type",
        type=str,
        default="absolute",
        choices=["absolute", "relative_base", "relative_delta"],
        help=(
            "absolute: actions[i] is absolute joint target.\n"
            "relative_base: actions[i] is offset from base qpos at start of 1Hz cycle.\n"
            "relative_delta: actions[i] is delta from previous executed target (integrate).\n"
            "NOTE: these are JOINT-space relative schemes. The VTAM relative_eef_rot6d mode is a\n"
            "SE(3) end-effector rot6d scheme handled entirely inside vtam_server.py (which composes\n"
            "the relative prediction back to an absolute pose6 before returning it); it is NOT one of\n"
            "these joint paths and must not be routed through them."
        ),
    )
    parser.add_argument("--initial_gripper_pos", type=int, default=None,
                        help="Initial gripper width (overrides INITIAL_GRIPPER_POS constant)")
    args = parser.parse_args()

    if args.initial_gripper_pos is not None:
        global INITIAL_GRIPPER_POS
        INITIAL_GRIPPER_POS = args.initial_gripper_pos

    infer = InferenceXArm(
        config_file=args.config_file,
        model_path=args.ckpt_path,
        device=args.device,
        exec_step=args.exec_step,
        threshold=args.threshold,
        num_inference_steps=args.num_inference_steps,
    )

    # robot init
    arm = XArmAPI(args.ip, do_not_open=True)
    arm.connect()
    arm.clean_gripper_error()
    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(4000)
    arm.set_gripper_position(INITIAL_GRIPPER_POS, wait=True)
    arm.motion_enable(True)
    arm.set_mode(1)
    arm.set_state(0)

    # cameras
    cap_base = cv2.VideoCapture(args.base_cam_index)
    cap_third = cv2.VideoCapture(args.third_cam_index) if args.use_third else None
    cap_gel = cv2.VideoCapture(args.gelsight_cam_index) if args.use_gelsight else None

    caps = [cap_base]
    if cap_third is not None:
        caps.append(cap_third)
    if cap_gel is not None:
        caps.append(cap_gel)

    for cap in caps:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap_base.isOpened():
        print("❌ BASE_CAM_INDEX 打开失败")
        return
    if cap_third is not None and not cap_third.isOpened():
        print("❌ THIRD_CAM_INDEX 打开失败")
        return
    if cap_gel is not None and not cap_gel.isOpened():
        print("❌ GELSIGHT_CAM_INDEX 打开失败")
        return

    # log book (CSV)
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pred_log_path = os.path.join(log_dir, f"pred_1hz_{timestamp}.csv")
    state_log_path = os.path.join(log_dir, f"state_30hz_{timestamp}.csv")
    pred_log_f = open(pred_log_path, "w", newline="")
    state_log_f = open(state_log_path, "w", newline="")
    pred_writer = csv.writer(pred_log_f)
    state_writer = csv.writer(state_log_f)
    pred_writer.writerow(
        ["time", "frame_idx", "step_idx", "action_type", "qpos0", "qpos1", "qpos2", "qpos3", "qpos4", "qpos5", "gpos",
         "pred_action0", "pred_action1", "pred_action2", "pred_action3", "pred_action4", "pred_action5", "pred_action6",
         "pred_state0", "pred_state1", "pred_state2", "pred_state3", "pred_state4", "pred_state5", "pred_state6"]
    )
    state_writer.writerow(
        ["time", "frame_idx", "step_idx", "qpos0", "qpos1", "qpos2", "qpos3", "qpos4", "qpos5", "gpos",
         "cmd0", "cmd1", "cmd2", "cmd3", "cmd4", "cmd5", "cmd_g"]
    )

    # Initialize GelSight force estimator (only used when state_dim == 19)
    state_dim = infer.states_min_np.shape[1]
    gel_force_est = None
    if state_dim == 19 and cap_gel is not None:
        gel_force_est = GelSightForceEstimator(baseline_n=FORCE_BASELINE_N)
        print(f"🔬 Capturing GelSight force baseline ({FORCE_BASELINE_N} frames, keep sensor unloaded)...")
        for _ in range(FORCE_BASELINE_N + 5):
            cap_gel.grab()
            ret_bl, img_bl = cap_gel.retrieve()
            if ret_bl:
                gel_force_est.update(img_bl)
            time.sleep(0.05)

    frame_idx = 0
    print(f"\n🤖 Manual-trigger mode | prompt: \"{args.prompt}\" | exec_step: {args.exec_step} @ {args.fps}Hz")
    print("   Press Enter to: capture → infer → execute.  Ctrl+C to quit.\n")

    try:
        while True:
            # ── Wait for keyboard trigger ──
            input(f">>> [Cycle {frame_idx}] Press Enter to run next inference cycle...")

            # ── 1. Capture images ──
            # flush camera buffers
            for _ in range(5):
                cap_base.grab()
                if cap_third is not None:
                    cap_third.grab()
                if cap_gel is not None:
                    cap_gel.grab()

            ret_b, img_b_full = cap_base.retrieve()
            if not ret_b:
                print("⚠️  base camera read failed, retry")
                continue

            img_t_full = None
            img_g_full = None
            if cap_third is not None:
                ret_t, img_t_full = cap_third.retrieve()
                if not ret_t:
                    print("⚠️  third camera read failed, retry")
                    continue
            if cap_gel is not None:
                ret_g, img_g_full = cap_gel.retrieve()
                if not ret_g:
                    print("⚠️  gelsight camera read failed, retry")
                    continue

            # Model input: raw RGB (play() handles resize + normalize with training transforms)
            ge_b = cv2.cvtColor(img_b_full, cv2.COLOR_BGR2RGB)
            target_h, target_w = ge_b.shape[:2]
            views = [ge_b]
            if img_t_full is not None:
                ge_t = cv2.cvtColor(img_t_full, cv2.COLOR_BGR2RGB)
                if ge_t.shape[:2] != (target_h, target_w):
                    ge_t = cv2.resize(ge_t, (target_w, target_h))
                views.append(ge_t)
            if img_g_full is not None and not args.gelsight_force_only:
                ge_g = cv2.cvtColor(img_g_full, cv2.COLOR_BGR2RGB)
                if ge_g.shape[:2] != (target_h, target_w):
                    ge_g = cv2.resize(ge_g, (target_w, target_h))
                views.append(ge_g)

            stacked_obs = np.stack(views, axis=0)  # V,H,W,3 uint8 RGB

            # ── 2. Read robot state ──
            code, qpos = arm.get_servo_angle(is_radian=True)
            _, gpos = arm.get_gripper_position()
            if code != 0 or gpos is None:
                print("⚠️  robot state read failed, retry")
                continue

            # Save debug mosaic
            img_b_proc = process_image(img_b_full)
            img_t_proc = process_image(img_t_full) if img_t_full is not None else None
            img_g_proc = process_image(img_g_full) if img_g_full is not None else None
            pane1 = img_b_proc.copy()
            pane2 = img_t_proc.copy() if img_t_proc is not None else np.zeros_like(img_b_proc)
            pane3 = img_g_proc.copy() if img_g_proc is not None else np.zeros_like(img_b_proc)
            cv2.putText(pane3, f"Prompt: {args.prompt}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            debug_img = np.hstack([pane1, pane2, pane3])
            cv2.imwrite("eval_xarm_input.jpg", debug_img)

            print(f"   State: qpos={[f'{q:.3f}' for q in qpos[:6]]}, gpos={gpos:.1f}")

            # ── 3. Build normalized state (same as training) ──
            state_dim = infer.states_min_np.shape[1]
            if state_dim == 19:
                # 19D: joint(6) + gripper(1) + tcp_xyz(3) + rot6d(6) + force(3)
                tcp_code, tcp_raw = arm.get_position(is_radian=True)
                if tcp_code != 0:
                    print("⚠️  TCP read failed, retry")
                    continue
                tcp_xyz = np.array(tcp_raw[:3], dtype=np.float32)  # mm
                rot_6d = euler_to_rotation_6d(tcp_raw[3], tcp_raw[4], tcp_raw[5])  # (6,)
                # Force: compute from GelSight optical flow
                if gel_force_est is not None and img_g_full is not None:
                    fx, fy, fz = gel_force_est.update(img_g_full)
                    force_xyz = np.array([fx, fy, fz], dtype=np.float32)
                else:
                    force_xyz = np.zeros(3, dtype=np.float32)
                current_state = np.concatenate([
                    np.array(qpos[:6], dtype=np.float32),  # 6D joint
                    np.array([gpos], dtype=np.float32),    # 1D gripper
                    tcp_xyz,                                # 3D TCP position
                    rot_6d,                                 # 6D rotation
                    force_xyz,                              # 3D force
                ], axis=0)[None, :]  # (1,19)
                print(f"   TCP: xyz=[{tcp_xyz[0]:.1f}, {tcp_xyz[1]:.1f}, {tcp_xyz[2]:.1f}]mm"
                      f"  Force: [{force_xyz[0]:.4f}, {force_xyz[1]:.4f}, {force_xyz[2]:.6f}]")
            elif state_dim == 16:
                # 16D: joint(6) + gripper(1) + tcp_xyz(3) + rot6d(6) — NO force in state
                tcp_code, tcp_raw = arm.get_position(is_radian=True)
                if tcp_code != 0:
                    print("⚠️  TCP read failed, retry")
                    continue
                tcp_xyz = np.array(tcp_raw[:3], dtype=np.float32)  # mm
                rot_6d = euler_to_rotation_6d(tcp_raw[3], tcp_raw[4], tcp_raw[5])  # (6,)
                current_state = np.concatenate([
                    np.array(qpos[:6], dtype=np.float32),  # 6D joint
                    np.array([gpos], dtype=np.float32),    # 1D gripper
                    tcp_xyz,                                # 3D TCP position
                    rot_6d,                                 # 6D rotation
                ], axis=0)[None, :]  # (1,16)
                print(f"   TCP: xyz=[{tcp_xyz[0]:.1f}, {tcp_xyz[1]:.1f}, {tcp_xyz[2]:.1f}]mm (no force in state)")
            else:
                # 7D: joint(6) + gripper(1)
                current_state = np.concatenate(
                [np.array(qpos[:6], dtype=np.float32), np.array([gpos], dtype=np.float32)], axis=0
            )[None, :]  # (1,7)

            state_norm = (current_state - infer.states_min_np) / (infer.states_max_np - infer.states_min_np + 1e-6)
            state_norm = state_norm * 2.0 - 1.0
            # action_dim = basic_action_dim(7) + state_dim; zeros pad the action part
            action_zeros = np.zeros((1, infer.basic_action_dim), dtype=np.float32)
            state_in = np.concatenate(
                [action_zeros, state_norm], axis=1
            ).astype(np.float32)  # (1, 7+state_dim) = (1, action_dim)

            # ── 4. Inference ──
            t_infer_start = time.time()
            actions_norm_last, actions_norm_full = infer.play(
                stacked_obs,
                args.prompt,
                execution_step=args.exec_step,
                state=state_in,
                return_normalized=True,
                return_full=True,
            )
            t_infer = time.time() - t_infer_start
            actions_norm_last = actions_norm_last.numpy()
            actions_norm_full = actions_norm_full.numpy()

            # Split and denormalize (basic_action_dim = 7 for normal, 10 for force_in_action)
            bad = infer.basic_action_dim
            pred_action_norm = actions_norm_full[:, :bad]
            pred_state_norm = actions_norm_full[:, bad:]
            pred_action_abs = (pred_action_norm + 1.0) / 2.0 * (infer.act_max_np - infer.act_min_np + 1e-6) + infer.act_min_np
            pred_state_abs = (pred_state_norm + 1.0) / 2.0 * (infer.states_max_np - infer.states_min_np + 1e-6) + infer.states_min_np

            if args.action_type == "absolute":
                actions = pred_state_abs
            elif args.action_type == "relative_base":
                action_norm = pred_action_norm.copy()
                action_norm[:, :6] = action_norm[:, :6] + state_norm[:, :6]
                actions = (action_norm + 1.0) / 2.0 * (infer.act_max_np - infer.act_min_np + 1e-6) + infer.act_min_np
            else:  # relative_delta
                integrated_norm = state_norm.copy()
                actions = np.zeros_like(pred_action_norm, dtype=np.float32)
                for k in range(pred_action_norm.shape[0]):
                    integrated_norm[:, :6] = integrated_norm[:, :6] + pred_action_norm[k:k+1, :6]
                    integrated_norm[:, 6:] = pred_action_norm[k:k+1, 6:]
                    actions[k] = ((integrated_norm + 1.0) / 2.0 * (infer.act_max_np - infer.act_min_np + 1e-6) + infer.act_min_np)[0]

            print(f"   Inference done in {t_infer:.2f}s | actions shape: {actions.shape}")
            print(f"   First action:  {actions[0]}")
            print(f"   Last action:   {actions[min(args.exec_step-1, actions.shape[0]-1)]}")

            # ── 5. Log predictions ──
            now_ts = time.time()
            for step_idx in range(pred_action_abs.shape[0]):
                pred_writer.writerow([
                    now_ts, frame_idx, step_idx, args.action_type,
                    *qpos[:6], gpos,
                    *pred_action_abs[step_idx].tolist(),
                    *pred_state_abs[step_idx].tolist(),
                ])
            pred_log_f.flush()

            # ── 6. Execute actions (identical to replay_xarm_episode.py) ──
            n_exec = min(args.exec_step, actions.shape[0])
            print(f"   Executing {n_exec} steps @ {args.fps}Hz ...")
            for i in range(n_exec):
                step_start = time.time()

                target_q = actions[i][:6].astype(np.float32)
                target_g = float(FIXED_GRIPPER_POS) if FIXED_GRIPPER_POS is not None else float(actions[i][6])

                # Direct send – no smoothing (identical to replay)
                arm.set_servo_angle_j(angles=target_q.tolist(), is_radian=True)

                if i % 3 == 0:
                    arm.clean_gripper_error()
                    arm.set_gripper_position(target_g, wait=False, speed=4000)

                # 30Hz log
                code_s, qpos_s = arm.get_servo_angle(is_radian=True)
                _, gpos_s = arm.get_gripper_position()
                if code_s == 0 and gpos_s is not None:
                    state_writer.writerow([
                        time.time(), frame_idx, i,
                        *np.array(qpos_s[:6]).tolist(), gpos_s,
                        *target_q.tolist(), target_g,
                    ])
                    state_log_f.flush()

                # Keep 30Hz
                elapsed = time.time() - step_start
                sleep_t = max(0.0, 1.0 / args.fps - elapsed)
                if sleep_t > 0:
                    time.sleep(sleep_t)

            print(f"   ✅ Cycle {frame_idx} done.\n")
            frame_idx += 1

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        arm.disconnect()
        pred_log_f.close()
        state_log_f.close()
        cap_base.release()
        if cap_third is not None:
            cap_third.release()
        if cap_gel is not None:
            cap_gel.release()


if __name__ == "__main__":
    main()
