# VTAM: Efficient Visuo-Tactile World-Action Models for Contact-Rich Manipulation without Tactile Pretraining

[[Paper]](https://arxiv.org/abs/2603.23481) [[Project page]](https://plan-lab.github.io/projects/vtam/)

VTAM fine-tunes a pretrained video world model to jointly predict future visual and tactile latents from task
demonstrations only (no tactile or action pretraining), then trains a flow-matching action expert end-to-end that
reads the world model's predictive latents in a single forward pass and predicts actions, proprioceptive states and
a light flow-force proxy computed from tactile optical flow.

## Release status

- [x] Training and inference code
- [x] Real-robot experiment configs and dataset statistics
- [x] Real-robot data (xArm6 + GelSight Mini): [JensenYuan/VTAM_xArm_GelSight](https://huggingface.co/datasets/JensenYuan/VTAM_xArm_GelSight)
- [ ] UniVTAC simulation configs
- [ ] Checkpoints

## Installation

```bash
git clone https://github.com/PLAN-Lab/VTAM.git
cd VTAM
pip install -r requirements.txt
```

Then fetch the upstream weights the configs expect: the Genie Envisioner video backbone, and the LTX-Video VAE,
tokenizer and text encoder it is built on.

```bash
hf download agibot-world/Genie-Envisioner-v1.0 GE_base_fast_v0.1.safetensors --local-dir pretrained_models

hf download Lightricks/LTX-Video --local-dir pretrained_models/ltx_video \
    --include "model_index.json" "vae/*" "tokenizer/*" "text_encoder/*"
```

With `<PRETRAINED_ROOT>=pretrained_models` this gives the layout the configs use:

```
pretrained_models/GE_base_fast_v0.1.safetensors
pretrained_models/ltx_video/{model_index.json,vae/,tokenizer/,text_encoder/}
```

`GE_base_fast_v0.1.safetensors` and the LTX-Video components are distributed under their own licenses (the LTX-Video
Open Weights License), not under this repository's licenses.

## Training

Training reads LeRobot-format datasets. Replace the `<DATA_ROOT>`, `<OUTPUT_ROOT>` and `<PRETRAINED_ROOT>` placeholders
in a task's configs, then run the two stages (shown for chip pick-and-place; we trained on 4 GPUs):

```bash
# Stage I: visuo-tactile world-model fine-tuning
torchrun --nproc_per_node=4 main.py --config_file configs/real_world/chip_pick_and_place/stage1_world_model.yaml

# Stage II: VTAM action learning, warm-started from a Stage I checkpoint (set model_path in the config)
torchrun --nproc_per_node=4 main.py --config_file configs/real_world/chip_pick_and_place/stage2_vtam.yaml
```

## Real-robot evaluation

`experiments/eval_xarm.py` runs a trained Stage II model on an xArm6 with RealSense cameras and a GelSight Mini, using a
task's `deploy_vtam.yaml` (it also needs the [xArm Python SDK](https://github.com/xArm-Developer/xArm-Python-SDK)):

```bash
python experiments/eval_xarm.py --config_file configs/real_world/chip_pick_and_place/deploy_vtam.yaml \
    --ckpt_path <stage2 checkpoint> --ip <xArm IP> --use_third --use_gelsight \
    --prompt "pick up the chip and place it on the plate"
```

Camera indices (`--base_cam_index`, `--third_cam_index`, `--gelsight_cam_index`) depend on your machine.
`vtam_server.py` serves a trained model over TCP for remote clients (used for our UniVTAC simulation evaluation).

## Real-robot configs

`configs/real_world/<task>/` holds, for each task:

| File | What it is |
|---|---|
| `stage1_world_model.yaml` | Stage I: task-specific visuo-tactile world-model fine-tuning, warm-started from `GE_base_fast_v0.1` |
| `stage2_vtam.yaml` | Stage II: VTAM action learning, warm-started from the Stage I checkpoint (paper result) |
| `deploy_vtam.yaml` | Real-robot inference config for the Stage II model (where available) |
| `ablation_tactile_vae_to_expert.yaml` | Table 5 ablation: frozen-VAE tactile features fed directly to the action expert (warm-started from a vision-only Stage I) |
| `stage1_world_model_vision_only.yaml` | The vision-only Stage I used by that ablation |
| `ablation_wo_flow_force_proxy.yaml` | Table 5 ablation: flow-force proxy removed from the action target (where available) |
| `stats.json`, `stats_world_model.json` | Normalization statistics used by the configs |

| Task | Robot / tactile sensor | Views | VTAM success |
|---|---|---|---|
| `chip_pick_and_place` | xArm6 / GelSight Mini | 2 RGB + tactile | 90% |
| `cucumber_peeling` | xArm6 / GelSight Mini | 2 RGB + tactile | 85% |
| `whiteboard_wiping` | xArm6 / GelSight Mini | 2 RGB + tactile | 95% |
| `hanoi_disk_insertion_xense` | Flexiv Rizon 4 / Xense Photon | 1 wrist RGB + tactile | 75% |

Notes:
- All paths are placeholders. Set `<DATA_ROOT>` (LeRobot-format datasets), `<OUTPUT_ROOT>` (training outputs) and
  `<PRETRAINED_ROOT>` (`GE_base_fast_v0.1.safetensors` and the LTX-Video components) for your setup.
- The tactile stream is called `gelsight` in every config, including Hanoi Disk Insertion, where it carries the
  Xense Photon images (the frozen video VAE encodes them zero-shot).
- Hanoi Disk Insertion uses a relative end-effector action (`relative_eef_rot6d`); the other tasks use absolute
  joint actions. In all tasks the last three action dimensions are the flow-force proxy.

## License

This repository is not under a single license; every source file states its license in its header, and
[NOTICE](NOTICE) maps it out.

- Original VTAM code, the configs, and the code from Diffusers / LTX-Video / openpi that Genie Envisioner redistributes
  (`models/ltx_models/`, `models/pipeline/`, `web_infer_utils/openpi_client/`) are under the
  [Apache License 2.0](LICENSES/Apache-2.0.txt).
- Files from Genie Envisioner outside those directories, and our modifications of them, stay under
  [CC BY-NC-SA 4.0](LICENSES/CC-BY-NC-SA-4.0.txt) because of its ShareAlike term, and **may not be used commercially**.

## Citation

```bibtex
@article{vtam2026,
  title={VTAM: Efficient Visuo-Tactile World-Action Models for Contact-Rich Manipulation without Tactile Pretraining},
  author={Yuan, Haoran and Yi, Weigang and Zhang, Zhenyu and Chen, Wendi and Mo, Yuchen and Yin, Jiashi and Li, Xinzhuo and Luo, Yue and Li, Jialin and Zeng, Xiangyu and Wen, Chuan and Lu, Cewu and Driggs-Campbell, Katherine and Lourentzou, Ismini},
  journal={arXiv preprint arXiv:2603.23481},
  year={2026}
}
```
