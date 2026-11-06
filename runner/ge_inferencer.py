# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# From Genie-Envisioner (AgibotTech); modified by the VTAM authors. Non-commercial use only.
import os, random, math
from pathlib import Path
from typing import Any, Dict, List

from datetime import datetime, timedelta
import argparse
import json
import importlib
# ----------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib
from yaml import load, dump, Loader, Dumper
import numpy as np
from tqdm import tqdm
import torch
from torch import distributed as dist
from einops import rearrange
from copy import deepcopy
import transformers
import logging

# ----------------------------------------------------
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

# ----------------------------------------------------
from utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model, count_model_parameters, unwrap_model

# ----------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
from utils import init_logging, import_custom_class, save_video
from utils.data_utils import get_latents, get_text_conditions, gen_noise_from_condition_frame_latent, randn_tensor, apply_color_jitter_to_video

from data.utils.statistics import StatisticInfo



class Inferencer:

    def __init__(self, config_file, output_dir=None, weight_dtype=torch.bfloat16, device="cuda:0", action_norm_type="meanstd") -> None:
        
        cd = load(open(config_file, "r"), Loader=Loader)
        args = argparse.Namespace(**cd)
        args.lr = float(args.lr)
        args.epsilon = float(args.epsilon)
        args.weight_decay = float(args.weight_decay)

        self.args = args

        if output_dir is not None:
            self.args.output_dir = output_dir

        if self.args.load_weights == False:
            print('You are not loading the pretrained weights, please check the code.')

        # Tokenizers
        self.tokenizer = None

        # Text encoders
        self.text_encoder = None

        # Denoisers
        self.diffusion_model = None
        self.unet = None

        # Autoencoders
        self.vae = None

        # Scheduler
        self.scheduler = None

        self.args.output_dir = Path(self.args.output_dir)
        self.args.output_dir.mkdir(parents=True, exist_ok=True)

        current_time = datetime.now()
        start_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        self.save_folder = os.path.join(self.args.output_dir, start_time)
        if getattr(self.args, "sub_folder", False):
            self.save_folder = os.path.join(self.args.output_dir, self.args.sub_folder)
        os.makedirs(self.save_folder, exist_ok=True)

        args_dict = vars(deepcopy(self.args))
        for k, v in args_dict.items():
            args_dict[k] = str(v)
        with open(os.path.join(self.save_folder, 'config.json'), "w") as file:
            json.dump(args_dict, file, indent=4, sort_keys=False)
        
        self.weight_dtype = weight_dtype
        self.device = device


        self.StatisticInfo = StatisticInfo
        if self.args.data['val'].get('stat_file', None) is not None:
            with open(self.args.data['val']['stat_file'], "r") as f:
                self.StatisticInfo = json.load(f)

        self.action_norm_type = action_norm_type

    def prepare_val_dataset(self) -> None:
        if not hasattr(self.args, "val_data_class"):
            self.args.val_data_class = self.args.train_data_class
        print(f"Validation Dataset: {self.args.val_data_class}")

        val_dataset_class = import_custom_class(
            self.args.val_data_class, self.args.val_data_class_path
        )
        
        self.args.data['val'].update({"fix_epiidx": 0, "fix_sidx":0, "fix_mem_idx":[0,0,0,0]})

        self.val_dataset = val_dataset_class(**self.args.data['val'])
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=1, shuffle=False,
        )


    def prepare_models(self,):

        print("Initializing models")
        device = self.device
        dtype = self.weight_dtype

        ### Load Tokenizer
        tokenizer_class = import_custom_class(
            self.args.tokenizer_class, getattr(self.args, "tokenizer_class_path", "transformers")
        )
        textenc_class = import_custom_class(
            self.args.textenc_class, getattr(self.args, "textenc_class_path", "transformers")
        )
        cond_models = load_condition_models(
            tokenizer_class, textenc_class,
            self.args.pretrained_model_name_or_path if not hasattr(self.args, "tokenizer_pretrained_model_name_or_path") else self.args.tokenizer_pretrained_model_name_or_path,
            load_weights=self.args.load_weights
        )
        self.tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
        self.text_encoder = text_encoder.to(device, dtype=dtype).eval()
        self.text_uncond = get_text_conditions(self.tokenizer, self.text_encoder, prompt="")
        self.uncond_prompt_embeds = self.text_uncond['prompt_embeds']
        self.uncond_prompt_attention_mask = self.text_uncond['prompt_attention_mask']

        ### Load VAE
        vae_class = import_custom_class(
            self.args.vae_class, getattr(self.args, "vae_class_path", "transformers")
        )
        if getattr(self.args, 'vae_path', False):
            self.vae = load_vae_models(vae_class, self.args.vae_path).to(device, dtype=dtype).eval()
        else:
            self.vae = load_latent_models(vae_class, self.args.pretrained_model_name_or_path)["vae"].to(device, dtype=dtype).eval()
        if isinstance(self.vae.latents_mean, List):
            self.vae.latents_mean = torch.FloatTensor(self.vae.latents_mean)
        if isinstance(self.vae.latents_std, List):
            self.vae.latents_std = torch.FloatTensor(self.vae.latents_std)
        if self.vae is not None:
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()
        self.SPATIAL_DOWN_RATIO = self.vae.spatial_compression_ratio
        self.TEMPORAL_DOWN_RATIO = self.vae.temporal_compression_ratio
        print(f'SPATIAL_DOWN_RATIO of VAE :{self.SPATIAL_DOWN_RATIO}')
        print(f'TEMPORAL_DOWN_RATIO of VAE :{self.TEMPORAL_DOWN_RATIO}')


        ### Load Diffusion Model
        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class, getattr(self.args, "diffusion_model_class_path", "transformers")
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.args.diffusion_model['model_path'],
            load_weights=self.args.load_weights and getattr(self.args, "load_diffusion_model_weights", True),
            **self.args.diffusion_model['config']
        ).to(device, dtype=dtype)
        total_params = count_model_parameters(self.diffusion_model)
        print(f'Total parameters for transformer model:{total_params}')


        ### Load Diffuser Scheduler
        diffusion_scheduler_class = import_custom_class(
            self.args.diffusion_scheduler_class, getattr(self.args, "diffusion_scheduler_class_path", "diffusers")
        )
        if hasattr(self.args, "diffusion_scheduler_args"):
            self.scheduler = diffusion_scheduler_class(**self.args.diffusion_scheduler_args)
        else:
            self.scheduler = diffusion_scheduler_class()

        ### Import Inference Pipeline Class
        self.pipeline_class = import_custom_class(
            self.args.pipeline_class, getattr(self.args, "pipeline_class_path", "diffusers")
        )

    def validate(
        self, model_save_dir, global_step, n_view=1, n_chunk_video=1, n_chunk_action=10, n_validation=1, domain_name="agibotworld",
        return_attn_weights=False,
    ):

        os.makedirs(model_save_dir,exist_ok=True)

        pipe = self.pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer, self.diffusion_model
        )

        assert(self.args.return_action | self.args.return_video)


        if self.args.return_action:
            n_chunk_video = 1
            action_type = self.args.data["train"]["action_type"]
            action_space = self.args.data["train"]["action_space"]
            

        if self.args.return_video:
            n_chunk_action = 1


        for i_validation in range(n_validation):
            
            self.val_dataloader.dataset.fix_epiidx = i_validation

            if self.args.return_action:
                self.val_dataloader.dataset.fix_sidx = 0
                self.val_dataloader.dataset.fix_mem_idx = [1 for _ in range(self.args.data['train']['n_previous'])]

                pd_actions_arr_all = None
                gt_actions_arr_all = None

            all_cross_attn = [] if return_attn_weights else None

            if self.args.return_action:
                ### openloop result visualization
                # predicted + GT actions are both action-only (action_out_channels); for
                # relative_eef_rot6d this is the 13D relative target, not the 23D concat.
                num_action_dims = self.args.diffusion_model["config"]["action_out_channels"]
                nrows = (num_action_dims + 1) // 2
                fig, axes = plt.subplots(nrows, 2, figsize=(20, nrows * 2.8), sharex=True)
                axes = axes.flatten()
                total_steps = n_chunk_action * self.args.data["train"]["action_chunk"]

            for i_chunk_action in range(n_chunk_action):

                batch = next(iter(self.val_dataloader))
                image = batch['video'][:,:,:,:self.args.data['train']['n_previous']]  # shape b,c,v,t,h,w 
                prompt = batch['caption']
                gt_video = batch['video']

                b, c, v, t, h, w = image.shape

                negative_prompt = ''

                batch_size = 1

                image = image[:batch_size]

                image = rearrange(image, 'b c v t h w -> (b v) c t h w')

                if getattr(self.args, "add_state", False):
                    history_action_state = batch["state"][:batch_size] 
                    if history_action_state.shape[1] > 1:
                        history_action_state = history_action_state[:, self.args.data['train']['n_previous']-1:self.args.data['train']['n_previous'], :]
                    history_action_state = history_action_state.contiguous() ### B, 1, C
                else:
                    history_action_state = None

                preds = pipe.infer(
                    image=image,
                    prompt=prompt[:batch_size],
                    negative_prompt=negative_prompt,
                    num_inference_steps=self.args.num_inference_step,
                    decode_timestep=0.03,
                    decode_noise_scale=0.025,
                    guidance_scale=1.0,
                    height=h,
                    width=w,
                    n_view=v,
                    return_action=self.args.return_action,
                    n_prev=self.args.data['train']['n_previous'],
                    chunk=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1,
                    return_video=self.args.return_video,
                    noise_seed=42,
                    action_chunk=self.args.data['train']['action_chunk'],
                    history_action_state = history_action_state,
                    pixel_wise_timestep = self.args.pixel_wise_timestep,
                    n_chunk=n_chunk_video,
                    # diffusion sample width = action_out_channels (action-only); pipeline zero-pads
                    # to the history width for the model forward only. Equals action_in in legacy modes.
                    action_dim=self.args.diffusion_model["config"]["action_out_channels"] if self.args.return_action else None,
                    return_attn_weights=return_attn_weights,
                )[0]

                if return_attn_weights and "cross_attn_weights" in preds:
                    all_cross_attn.append({
                        "chunk_idx": i_chunk_action,
                        "weights": [w.cpu().float() for w in preds["cross_attn_weights"]],
                    })

                save_cap = f'Validation_{i_validation}'

                if self.args.return_video:
                    
                    video = preds['video'].data.cpu()

                    save_video(rearrange(gt_video[0].data.cpu(), 'c v t h w -> c t h (v w)', v=n_view), os.path.join(model_save_dir, f'{save_cap}_gt.mp4'), fps=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1)

                    save_video(rearrange(video, '(b v) c t h w -> b c t h (v w)', v=n_view)[0], os.path.join(model_save_dir, f'{save_cap}.mp4'), fps=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1)


                if self.args.return_action:

                    gt_actions = batch['actions'][:,self.args.data['train']['n_previous']:]

                    # shape t, c
                    pd_actions_arr = preds['action'][0].data.cpu().to(torch.float).numpy()
                    gt_actions_arr = gt_actions[0].data.cpu().to(torch.float).numpy()

                    # n_dim = pd_actions_arr.shape[-1]
                    
                    if pd_actions_arr_all is None:
                        pd_actions_arr_all = pd_actions_arr
                    else:
                        pd_actions_arr_all = np.concatenate((pd_actions_arr_all, pd_actions_arr), axis=0)
                    
                    if gt_actions_arr_all is None:
                        gt_actions_arr_all = gt_actions_arr
                    else:
                        gt_actions_arr_all = np.concatenate((gt_actions_arr_all, gt_actions_arr), axis=0)


                image = None

                ### prepare for next chunk action prediction
                self.val_dataloader.dataset.fix_sidx += self.args.data['train']['action_chunk']
                self.val_dataloader.dataset.fix_mem_idx = x = (np.linspace(0, self.val_dataloader.dataset.fix_sidx-1, self.args.data['train']['n_previous']).round().astype(np.int16)).tolist()


            if self.args.return_action:
                x_axis = np.arange(gt_actions_arr_all.shape[0])
                num_dims = gt_actions_arr_all.shape[-1]
                for dim_idx in range(num_dims):
                    
                    ax = axes[dim_idx]

                    # Plot the continuous action sequences
                    ax.plot(x_axis, gt_actions_arr_all[:, dim_idx], label='Ground Truth', color='cornflowerblue', alpha=0.9)
                    ax.plot(x_axis, pd_actions_arr_all[:, dim_idx], label='Inferred', color='tomato', linestyle='--', alpha=0.9)

                    # Mark the starting point of each inference sequence
                    start_indices = np.arange(0, gt_actions_arr_all.shape[0], self.args.data["train"]["action_chunk"])

                    ax.scatter(start_indices, gt_actions_arr_all[start_indices, dim_idx], c='blue', marker='o', s=40, zorder=5, label='GT Start')

                    ax.scatter(start_indices, pd_actions_arr_all[start_indices, dim_idx], c='darkred', marker='x', s=40, zorder=5, label='Inferred Start')

                    # ax.set_title(f'Action Dimension {dim_idx}')
                    ax.set_title(f"Dimension- {dim_idx}")

                ax.set_ylabel('Value')
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend()

                # Set common X-axis label
                fig.supxlabel(f'Continuous Timestep (across {n_chunk_action} inferences)')
                
                plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
                fig.suptitle(f'Comparison of Ground Truth and Inferred Actions', fontsize=18)
                
                plt.savefig(f'{self.save_folder}/openloop_evaluation_val{i_validation}.png', dpi=300, bbox_inches='tight')
                plt.clf()

            if return_attn_weights and all_cross_attn:
                valid_cam = self.args.data["train"]["valid_cam"]
                self._save_cross_attention_analysis(
                    all_cross_attn, model_save_dir, i_validation, valid_cam
                )


    @staticmethod
    def _smooth(values, window=10):
        """Sliding-window average for smoothing curves."""
        if len(values) <= window:
            return values
        kernel = np.ones(window) / window
        padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    def _save_cross_attention_analysis(self, all_cross_attn, save_dir, val_idx, valid_cam):
        """
        Per-view attention weight curves over time.

        Weights are taken post-softmax so each step's per-view values sum to 1.
        Produces:
          1. Per-layer plots: 28 subplots, each with n_views curves (sum=1 per step).
          2. All-layer-averaged plot: averaged across 28 layers (sum=1 per step).
        """
        attn_dir = os.path.join(save_dir, f"cross_attn_val{val_idx}")
        os.makedirs(attn_dir, exist_ok=True)

        n_chunks = len(all_cross_attn)
        n_layers = len(all_cross_attn[0]["weights"])
        n_views = len(valid_cam)

        # The raw post-softmax attention tensors (all heads/queries/tokens x 28
        # layers x chunks) are large and only needed for bespoke offline
        # re-analysis; the per-view summary PNGs below cover the normal case.
        # Off by default -- set save_raw_attn_weights: true in the config to dump it.
        if getattr(self.args, "save_raw_attn_weights", False):
            torch.save(
                {f"chunk_{c['chunk_idx']}": c["weights"] for c in all_cross_attn},
                os.path.join(attn_dir, "raw_weights.pt"),
            )

        # per_layer_view_attn[layer][step] = (n_views,)  post-softmax, sums to 1
        per_layer_view_attn = [[None] * n_chunks for _ in range(n_layers)]

        for step_idx, c in enumerate(all_cross_attn):
            for layer_idx in range(n_layers):
                w = c["weights"][layer_idx]  # (B, H, Q_action, K_video)  post-softmax
                # avg over heads & action queries → (K_video,)
                w_avg = w[0].mean(dim=0).mean(dim=0)
                tokens_per_view = w_avg.shape[0] // n_views

                view_vals = np.array([
                    w_avg[v * tokens_per_view : (v + 1) * tokens_per_view].sum().item()
                    for v in range(n_views)
                ])
                per_layer_view_attn[layer_idx][step_idx] = view_vals

        # Stack into arrays: (n_layers, n_chunks, n_views)
        attn_array = np.array(per_layer_view_attn)

        # All-layer average (preserves sum=1 since each layer sums to 1)
        avg_across_layers = attn_array.mean(axis=0)  # (n_chunks, n_views)

        smooth_window = min(10, max(1, n_chunks // 3))
        steps = np.arange(n_chunks)
        view_colors = plt.cm.tab10(np.linspace(0, 1, max(n_views, 10)))[:n_views]

        # ---- Plot 1: Averaged across all layers ----
        fig, ax = plt.subplots(figsize=(8, 4))
        for v_idx in range(n_views):
            ax.plot(steps, self._smooth(avg_across_layers[:, v_idx], smooth_window),
                    label=valid_cam[v_idx], color=view_colors[v_idx], linewidth=2)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Attention")
        ax.set_title("Cross-Attention per View (averaged over all layers)")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(attn_dir, "attn_per_view_all_layers.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # ---- Plot 2: Per-layer subplots (28 panels) ----
        n_cols = 4
        n_rows = math.ceil(n_layers / n_cols)
        fig_layers, axes_layers = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3))
        axes_layers = axes_layers.flatten()

        for layer_idx in range(n_layers):
            ax = axes_layers[layer_idx]
            for v_idx in range(n_views):
                curve = attn_array[layer_idx, :, v_idx]
                ax.plot(steps, self._smooth(curve, smooth_window),
                        label=valid_cam[v_idx] if layer_idx == 0 else None,
                        color=view_colors[v_idx], linewidth=1.5)
            ax.set_title(f"Layer {layer_idx}", fontsize=9)
            ax.set_xlabel("Steps", fontsize=7)
            ax.set_ylabel("Attention", fontsize=7)
            ax.tick_params(labelsize=6)

        for idx in range(n_layers, len(axes_layers)):
            axes_layers[idx].axis("off")

        handles = [plt.Line2D([], [], color=view_colors[v], linewidth=2, label=valid_cam[v])
                   for v in range(n_views)]
        fig_layers.legend(handles=handles, loc="upper center", ncol=n_views, fontsize=9,
                          bbox_to_anchor=(0.5, 1.0))
        fig_layers.suptitle("Per-Layer Cross-Attention per View", fontsize=14, y=1.03)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(attn_dir, "attn_per_view_per_layer.png"), dpi=200, bbox_inches="tight")
        plt.close(fig_layers)

        # ---- Save numerical data ----
        summary = {
            "n_layers": n_layers,
            "n_steps": n_chunks,
            "n_views": n_views,
            "valid_cam": valid_cam,
            "smooth_window": smooth_window,
            "avg_all_layers": {cam: avg_across_layers[:, i].tolist() for i, cam in enumerate(valid_cam)},
            "per_layer": {
                f"layer_{l}": {cam: attn_array[l, :, i].tolist() for i, cam in enumerate(valid_cam)}
                for l in range(n_layers)
            },
        }
        with open(os.path.join(attn_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[Cross-Attention] Saved {n_views}-view analysis to {attn_dir}")


    def infer(self, n_chunk_action=4, n_chunk_video=1, n_validation=10, global_step=0, domain_name="agibotworld", return_attn_weights=False):
        model_save_dir = os.path.join(self.save_folder,f'Inference')
        self.validate(
            model_save_dir, global_step,
            n_view=len(self.args.data["train"]["valid_cam"]),
            n_chunk_video=n_chunk_video,
            n_chunk_action=n_chunk_action,
            n_validation=n_validation,
            domain_name=domain_name,
            return_attn_weights=return_attn_weights,
        )
