# VTAM: Efficient Visuo-Tactile World-Action Models for Contact-Rich Manipulation without Tactile Pretraining

[[Paper]](https://arxiv.org/abs/2603.23481) [[Project page]](https://plan-lab.github.io/projects/vtam/)

VTAM fine-tunes a pretrained video world model to jointly predict future visual and tactile latents from task
demonstrations only (no tactile or action pretraining), then trains a flow-matching action expert end-to-end that
reads the world model's predictive latents in a single forward pass and predicts actions, proprioceptive states and
a light flow-force proxy computed from tactile optical flow.

## Release status

- [x] Real-robot experiment configs and dataset statistics (this release)
- [ ] Training and inference code
- [ ] UniVTAC simulation configs
- [ ] Checkpoints and datasets

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

The files in this release are under the Apache License 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).

## Citation

```bibtex
@article{vtam2026,
  title={VTAM: Efficient Visuo-Tactile World-Action Models for Contact-Rich Manipulation without Tactile Pretraining},
  author={Yuan, Haoran and Yi, Weigang and Zhang, Zhenyu and Chen, Wendi and Mo, Yuchen and Yin, Jiashi and Li, Xinzhuo and Luo, Yue and Li, Jialin and Zeng, Xiangyu and Wen, Chuan and Lu, Cewu and Driggs-Campbell, Katherine and Lourentzou, Ismini},
  journal={arXiv preprint arXiv:2603.23481},
  year={2026}
}
```
