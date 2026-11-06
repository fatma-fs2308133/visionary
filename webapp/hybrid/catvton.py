"""Minimal CatVTON inference for Apple MPS / CPU.

Re-implements the inference path of CatVTONPipeline
(github.com/Zheng-Chong/CatVTON, model/pipeline.py and model/utils.py,
CC BY-NC-SA 4.0) with three differences:
  - device "mps" (or "cpu"), fp16 UNet, no CUDA calls, no safety checker;
  - the mask is supplied by us (our pose + DLT warp + parsing) instead of
    CatVTON's DensePose + SCHP AutoMasker (detectron2 does not build on macOS);
  - optional warm start: begin denoising from a noised version of an initial
    image (our classical composite) instead of pure noise ("strength" < 1,
    as in SDEdit).

Model: SD-1.5 inpainting UNet with cross-attention removed; only the
self-attention ("attn1") weights are replaced by CatVTON's trained ones
(vitonhd-16k-512, 49.6M params). Person and garment latents are concatenated
along the height axis so self-attention can match garment to body.
"""

import time

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0, SlicedAttnProcessor
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

BASE = "stable-diffusion-v1-5/stable-diffusion-inpainting"
VAE = "stabilityai/sd-vae-ft-mse"
REPO = "zhengchong/CatVTON"
CHECKPOINTS = {"vitonhd": "vitonhd-16k-512/attention",   # 512x384, VITON-HD
               "mix": "mix-48k-1024/attention"}          # 1024x768, VITON-HD + DressCode + DeepFashion


class SkipAttnProcessor(torch.nn.Module):
    """Cross-attention is removed: return the input unchanged."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        return hidden_states


class CatVTON:
    def __init__(self, device: str | None = None, dtype=torch.float16, checkpoint: str = "vitonhd",
                 attention: str = "sliced", slice_size: int = 4):
        """attention: "sliced" (fp32 scores, a few heads at a time), "upcast"
        (fp32 scores, all heads at once) or "sdpa" (PyTorch fused, fp16).
        Measured on an M1 Pro at 512x384, 25 steps: sliced 31.8 s / 5.5 GB,
        upcast 35.7 s / 9.9 GB, sdpa ~75 s; 1024x768 needs one 18 GiB buffer
        with sdpa and does not fit."""
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = dtype if self.device != "cpu" else torch.float32
        base = snapshot_download(BASE, allow_patterns=["unet/*", "scheduler/*", "*.json"])
        self.scheduler = DDIMScheduler.from_pretrained(base, subfolder="scheduler")
        # The VAE stays fp32: fp16 SD VAEs overflow and give black/NaN images
        self.vae = AutoencoderKL.from_pretrained(VAE).to(self.device, torch.float32)
        self.unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", variant="fp16",
                                                         torch_dtype=self.dtype).to(self.device)
        def self_proc():
            if attention == "sliced":
                return SlicedAttnProcessor(slice_size)
            return AttnProcessor() if attention == "upcast" else AttnProcessor2_0()

        procs = {name: self_proc() if name.endswith("attn1.processor") else SkipAttnProcessor()
                 for name in self.unet.attn_processors}
        self.unet.set_attn_processor(procs)
        # CatVTON saves get_trainable_module(unet, "attention"): every module
        # whose name contains "attn1". With its diffusers version that is 8
        # entries per block (attn1, to_q, to_k, to_v, to_out, to_out.0,
        # to_out.1, processor), so block k is stored under index 8k.
        blocks = [m for n, m in self.unet.named_modules() if n.endswith("attn1")]
        sub = CHECKPOINTS[checkpoint]
        self.checkpoint = checkpoint
        path = snapshot_download(REPO, allow_patterns=[f"{sub}/*"])
        state = load_file(f"{path}/{sub}/model.safetensors")
        per_block = {}
        for key, val in state.items():
            idx, rest = key.split(".", 1)
            if int(idx) % 8:
                raise RuntimeError(f"unexpected CatVTON key {key}")
            per_block.setdefault(int(idx) // 8, {})[rest] = val
        if sorted(per_block) != list(range(len(blocks))):
            raise RuntimeError(f"CatVTON has {len(per_block)} attention blocks, the UNet {len(blocks)}")
        for k, block in enumerate(blocks):
            block.load_state_dict(per_block[k], strict=True)
            block.to(self.device, self.dtype)
            block.upcast_attention = block.upcast_softmax = attention != "sdpa"

    # -------------------------------------------------------------------
    def _encode(self, img: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 127.5 - 1
        with torch.no_grad():
            lat = self.vae.encode(x.to(self.device)).latent_dist.sample()
        return lat * self.vae.config.scaling_factor

    @torch.no_grad()
    def __call__(self, person: np.ndarray, garment: np.ndarray, mask: np.ndarray, steps: int = 30,
                 guidance: float = 2.5, seed: int = 42, init: np.ndarray | None = None,
                 strength: float = 1.0, progress=None, anchor: np.ndarray | None = None,
                 anchor_until: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """person, garment, init: H x W x 3 uint8 (same size, multiples of 8;
        garment padded on white). mask: H x W bool (True = repaint).
        anchor / anchor_until: layout constraint (blended latent diffusion).
        anchor_until is an H x W map of step fractions in [0, 1]: while
        step / steps < anchor_until, the latents there are replaced by the
        anchor image noised to the current level, so the region keeps the
        anchor's layout (e.g. trousers where a cropped garment ends); the
        remaining steps refine it freely. 0 = unconstrained.
        Returns (output uint8, seconds)."""
        t0 = time.perf_counter()
        gen = torch.Generator("cpu").manual_seed(seed)
        m = torch.from_numpy(mask.astype(np.float32))[None, None].to(self.device)
        # CatVTON masks AFTER normalising to [-1, 1]: masked pixels are 0 =
        # mid grey (127.5), not black
        masked = np.where(mask[..., None], 127.5, person.astype(np.float32))
        masked_lat = self._encode(masked)
        garm_lat = self._encode(garment)
        mask_lat = torch.nn.functional.interpolate(m, size=masked_lat.shape[-2:], mode="nearest")

        cat = -2                                             # concatenate along height
        masked_cat = torch.cat([masked_lat, garm_lat], dim=cat)
        mask_cat = torch.cat([mask_lat, torch.zeros_like(mask_lat)], dim=cat)
        # classifier-free guidance: the unconditional branch sees no garment
        masked_cat = torch.cat([torch.cat([masked_lat, torch.zeros_like(garm_lat)], dim=cat), masked_cat])
        mask_cat = torch.cat([mask_cat] * 2)

        self.scheduler.set_timesteps(steps, device=self.device)
        timesteps = self.scheduler.timesteps
        noise = torch.randn(masked_cat[:1].shape, generator=gen).to(self.device)
        if init is not None and strength < 1:
            # warm start (SDEdit): noise our composite to the level of step k
            k = int(round(steps * (1 - strength)))
            timesteps = timesteps[k:]
            init_lat = torch.cat([self._encode(init), garm_lat], dim=cat)
            latents = self.scheduler.add_noise(init_lat, noise, timesteps[:1])
        else:
            latents = noise * self.scheduler.init_noise_sigma

        if anchor is not None:
            a_lat = torch.cat([self._encode(anchor), garm_lat], dim=cat)
            until = torch.from_numpy(anchor_until.astype(np.float32))[None, None].to(self.device)
            until = torch.nn.functional.interpolate(until, size=masked_lat.shape[-2:], mode="area")
            until = torch.cat([until, torch.zeros_like(until)], dim=cat)
            a_m = (until > 0).float()
            latents = a_m * self.scheduler.add_noise(a_lat, noise, timesteps[:1]) + (1 - a_m) * latents

        for i, t in enumerate(timesteps):
            if progress:
                progress(i / len(timesteps))
            inp = self.scheduler.scale_model_input(torch.cat([latents] * 2), t)
            inp = torch.cat([inp, mask_cat, masked_cat], dim=1).to(self.dtype)
            eps = self.unet(inp, t, encoder_hidden_states=None, return_dict=False)[0].float()
            uncond, cond = eps.chunk(2)
            eps = uncond + guidance * (cond - uncond)
            latents = self.scheduler.step(eps, t, latents, eta=1.0, generator=gen).prev_sample
            if anchor is not None and i + 1 < len(timesteps):
                a_m = (until > (i + 1) / len(timesteps)).float()
                ref = self.scheduler.add_noise(a_lat, noise, timesteps[i + 1:i + 2])
                latents = a_m * ref + (1 - a_m) * latents

        latents = latents.split(latents.shape[cat] // 2, dim=cat)[0] / self.vae.config.scaling_factor
        img = self.vae.decode(latents.float()).sample
        img = ((img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
        if self.device == "mps":
            torch.mps.synchronize()
        return img, time.perf_counter() - t0
