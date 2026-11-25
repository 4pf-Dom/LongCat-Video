from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Iterator

import gradio as gr
import torch
from huggingface_hub import snapshot_download


# This file lives at the root of the LongCat-Video repository. The same file can
# be copied to the root of a Hugging Face Space created from this repository.
ROOT = Path(__file__).resolve().parent
AVATAR_REPO = "meituan-longcat/LongCat-Video-Avatar-1.6"
BASE_REPO = "meituan-longcat/LongCat-Video"
AVATAR_DIR = ROOT / "weights" / "LongCat-Video-Avatar-1.6"
BASE_DIR = ROOT / "weights" / "LongCat-Video"
INFERENCE_SCRIPT = ROOT / "run_demo_avatar_single_audio_to_video.py"


def _download_models() -> tuple[Path, Path]:
    """Fetch the official avatar and shared base checkpoints on first run."""
    token = os.environ.get("HF_TOKEN") or None
    snapshot_download(AVATAR_REPO, local_dir=str(AVATAR_DIR), token=token)
    snapshot_download(BASE_REPO, local_dir=str(BASE_DIR), token=token)
    return AVATAR_DIR, BASE_DIR


def generate(
    audio_path: str | None,
    image_path: str | None,
    prompt: str,
    resolution: str,
    num_segments: int,
    seed: int,
) -> Iterator[tuple[str | None, str]]:
    if not audio_path:
        yield None, "Upload an audio clip to continue."
        return
    if not prompt or not prompt.strip():
        yield None, "Add a scene description in the prompt field."
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        yield None, (
            "This official inference path requires two CUDA GPUs for context parallelism. "
            "Use a Hugging Face Space configured with suitable multi-GPU hardware (for example, 2×L40S)."
        )
        return
    if not INFERENCE_SCRIPT.is_file():
        yield None, f"Inference script not found at {INFERENCE_SCRIPT}. Deploy this app with the repository code."
        return

    yield None, "Downloading model checkpoints if needed. The first run may take a while…"
    try:
        avatar_dir, _ = _download_models()
    except Exception as exc:
        yield None, f"Model download failed: {exc}"
        return

    run_id = uuid.uuid4().hex
    work_dir = Path(tempfile.mkdtemp(prefix=f"longcat-{run_id}-", dir=str(ROOT / "outputs")))
    inputs_dir = work_dir / "inputs"
    output_dir = work_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        audio_ext = Path(audio_path).suffix.lower()
        if audio_ext not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
            audio_ext = ".wav"
        audio_file = inputs_dir / f"speech{audio_ext}"
        shutil.copyfile(audio_path, audio_file)

        image_file: Path | None = None
        if image_path:
            image_ext = Path(image_path).suffix.lower()
            if image_ext not in {".png", ".jpg", ".jpeg", ".webp"}:
                image_ext = ".png"
            image_file = inputs_dir / f"reference{image_ext}"
            shutil.copyfile(image_path, image_file)

        input_json = inputs_dir / "input.json"
        payload = {
            "prompt": prompt.strip(),
            "cond_audio": {"person1": str(audio_file)},
        }
        stage = "at2v"
        if image_file:
            payload["cond_image"] = str(image_file)
            stage = "ai2v"
        input_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        command = [
            "torchrun", "--standalone", "--nproc_per_node=2", str(INFERENCE_SCRIPT),
            f"--checkpoint_dir={avatar_dir}",
            f"--input_json={input_json}",
            f"--output_dir={output_dir}",
            f"--stage_1={stage}",
            f"--resolution={resolution}",
            f"--num_segments={int(num_segments)}",
            "--context_parallel_size=2",
            "--model_type=avatar-v1.6",
            "--use_distill",
            "--lora_path=loraB/lora.safetensors",
            "--mix_steps=4",
            "--mix_alpha=1.0",
            f"--seed={int(seed)}",
        ]
        # The upstream Avatar script currently fixes its seed internally; preserve
        # the selected value as the process seed for deterministic initialization.
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(int(seed))
        env["PYTHONUNBUFFERED"] = "1"

        yield None, "Generating video with LongCat-Video-Avatar 1.6. This can take several minutes…"
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        tail: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            tail.append(line.rstrip())
            tail = tail[-16:]
            yield None, "Inference running…\n" + "\n".join(tail)
        return_code = process.wait()
        if return_code != 0:
            yield None, "Inference failed. Recent log output:\n" + "\n".join(tail)
            return

        videos = sorted(output_dir.rglob("*.mp4"), key=lambda path: path.stat().st_mtime)
        if not videos:
            yield None, "Inference finished, but no MP4 was found. Check the Space logs for details."
            return
        yield str(videos[-1]), "Done — your generated video is ready."
    except Exception as exc:
        yield None, f"Generation failed: {exc}"
    finally:
        # Keep generated video available to Gradio until it has copied it to its
        # managed cache; remove inputs and temporary logs only after the request.
        pass


with gr.Blocks(title="LongCat-Video-Avatar 1.6", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# LongCat-Video-Avatar 1.6\n"
        "Generate an audio-driven talking video from a reference image, or generate "
        "a character video from audio and a text description. This demo invokes the "
        "official repository inference script and model weights."
    )
    gr.Markdown(
        "**Hardware:** two CUDA GPUs required. First generation downloads the Avatar 1.6 "
        "and shared LongCat-Video checkpoints; it may take a long time and requires substantial disk space."
    )
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(label="Speech / singing audio", type="filepath", sources=["upload"])
            image = gr.Image(label="Optional reference image (image-to-video)", type="filepath")
            prompt = gr.Textbox(
                label="Scene prompt",
                lines=4,
                placeholder="Describe the person, clothing, setting, camera, and actions. Include that they are speaking or singing.",
                value="A person speaks naturally to the camera in a softly lit studio, expressive facial and head movements, static camera.",
            )
            with gr.Row():
                resolution = gr.Dropdown(["480p", "720p"], value="480p", label="Resolution")
                segments = gr.Slider(1, 3, value=1, step=1, label="Video segments (longer = slower)")
            seed = gr.Number(value=42, precision=0, label="Seed")
            submit = gr.Button("Generate video", variant="primary")
        with gr.Column():
            result = gr.Video(label="Generated video")
            status = gr.Textbox(label="Status", lines=7, interactive=False)
    submit.click(generate, [audio, image, prompt, resolution, segments, seed], [result, status])
    gr.Markdown(
        "Model: [LongCat-Video-Avatar-1.6](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6) · "
        "[Project page](https://meigen-ai.github.io/LongCat-Video-Avatar-1.6-Page/) · "
        "[Source repository](https://github.com/meituan-longcat/LongCat-Video)."
    )

if __name__ == "__main__":
    # App can run at repo root or from an uploaded Space root.
    (ROOT / "outputs").mkdir(exist_ok=True)
    demo.queue(max_size=1).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
