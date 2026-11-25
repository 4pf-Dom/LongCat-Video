---
title: LongCat-Video-Avatar 1.6 Demo
emoji: 🎬
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: 3.10
suggested_hardware: 2xL40S
suggested_storage: large
---

# LongCat-Video-Avatar 1.6 Space

This Space wraps the official single-audio-to-video inference script in this repository. Upload an audio clip and optionally a reference image, enter a scene prompt, and generate a talking/singing video. Without a reference image, it uses the audio-text-to-video stage.

## Hardware and first run

The upstream Avatar 1.6 demo uses context parallelism and this app launches it with **two CUDA GPUs**. Choose a multi-GPU Space hardware tier that provides two compatible GPUs; the suggested `2xL40S` tier is a practical starting point. Verify current availability and pricing in Hugging Face Spaces before selecting hardware. A CPU Space or a single GPU will not run this configuration.

The first generation downloads both `meituan-longcat/LongCat-Video-Avatar-1.6` and the shared `meituan-longcat/LongCat-Video` checkpoints. These are large model repositories; allow substantial persistent storage and a long cold start. Set `HF_TOKEN` as a Space secret only if your account needs it to access the model repositories.

## Deploy from this GitHub repository

The repository's main `README.md` is the upstream project README, so it is intentionally left unchanged. To create a Space from the demo:

1. Create a new Hugging Face **Gradio** Space and select suitable two-GPU hardware.
2. Copy this repository's `app.py` and `requirements-space.txt` to the Space root (rename the requirements file to `requirements.txt`).
3. Copy the full LongCat-Video repository source into the Space root as well, including `run_demo_avatar_single_audio_to_video.py` and the `longcat_video/` package. This app deliberately calls the official script instead of reimplementing model inference.
4. Copy this file to the Space root as `README.md` so the Space SDK metadata is applied.
5. In the Space Docker setup, install CUDA PyTorch 2.6.0 for CUDA 12.4 and FlashAttention 2.7.4.post1 before the Python requirements. The upstream instructions use Python 3.10 and CUDA 12.4; FlashAttention must match the selected PyTorch/CUDA image. Install system `ffmpeg` as well.

The inference runner uses Avatar 1.6's distillation LoRA (`loraB/lora.safetensors`), its 8-step distilled path, 480p by default, and one segment by default. Each extra segment increases runtime and memory requirements. The model weights and code are governed by their respective upstream licenses; review them before public deployment.
