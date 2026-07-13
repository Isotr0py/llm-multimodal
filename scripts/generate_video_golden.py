#!/usr/bin/env python3
"""Generate Qwen end-to-end video golden fixtures with Transformers."""

import argparse
import json
from pathlib import Path
from types import MethodType

import av
import numpy as np
from transformers import AutoProcessor
from transformers.video_utils import load_video

FRAME_COUNT, FPS, WIDTH, HEIGHT, SAMPLE_FPS = 30, 5, 96, 64, 2


MODEL_IDS = {
    "qwen2_vl": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen3_vl": "Qwen/Qwen3-VL-2B-Instruct",
}


def fetch_videos_with_pyav(self, video_or_videos, sample_indices_fn=None):
    """Use PyAV when TorchCodec is installed but lacks shared libraries."""
    if isinstance(video_or_videos, list):
        return list(
            zip(
                *[
                    self.fetch_videos(video, sample_indices_fn=sample_indices_fn)
                    for video in video_or_videos
                ]
            )
        )
    return load_video(
        video_or_videos,
        backend="pyav",
        sample_indices_fn=sample_indices_fn,
    )


def generate_source_video(path: Path) -> None:
    """Create a small deterministic complete H.264 video."""
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    stream.options = {"crf": "12", "preset": "veryslow"}
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    for frame_index in range(FRAME_COUNT):
        pixels = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
        pixels[..., 0] = (x * 3 + frame_index * 17) % 256
        pixels[..., 1] = (y * 4 + frame_index * 29) % 256
        pixels[..., 2] = ((x + y) * 2 + frame_index * 43) % 256
        left = 4 + frame_index % 24
        pixels[4:16, left : left + 24] = (
            (frame_index * 31) % 256,
            255 - (frame_index * 7) % 256,
            (frame_index * 13) % 256,
        )
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def rust_processor_config(processor) -> dict:
    size = processor.size
    return {
        "do_convert_rgb": processor.do_convert_rgb,
        "do_normalize": processor.do_normalize,
        "do_rescale": processor.do_rescale,
        "do_resize": processor.do_resize,
        "image_mean": list(processor.image_mean),
        "image_std": list(processor.image_std),
        "rescale_factor": processor.rescale_factor,
        "resample": int(processor.resample),
        "patch_size": processor.patch_size,
        "merge_size": processor.merge_size,
        "temporal_patch_size": processor.temporal_patch_size,
        "min_pixels": int(size["shortest_edge"]),
        "max_pixels": int(size["longest_edge"]),
    }


def generate_model_golden(model: str, video_path: Path, output_dir: Path) -> None:
    processor = AutoProcessor.from_pretrained(MODEL_IDS[model])
    video_processor = processor.video_processor
    video_processor.fetch_videos = MethodType(fetch_videos_with_pyav, video_processor)
    conversation = [
        {
            "role": "user",
            "content": [
                # Use the chat-template URL schema with a local absolute path,
                # keeping fixture generation independent of external hosting.
                {"type": "video", "url": str(video_path.resolve())},
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    # Start at the chat/video-URL boundary. Transformers resolves the path,
    # decodes the video, invokes the model sampler, and returns model inputs.
    outputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="np",
        processor_kwargs={
            "do_sample_frames": True,
            "fps": SAMPLE_FPS,
            "return_metadata": True,
        },
    )
    metadata = outputs["video_metadata"][0]
    indices = np.asarray(metadata.frames_indices, dtype=np.int64)
    pixels = np.asarray(outputs["pixel_values_videos"], dtype=np.float32)
    grid = np.asarray(outputs["video_grid_thw"], dtype=np.int64)
    np.savez_compressed(
        output_dir / f"golden_{model}.npz",
        pixel_values=pixels,
        video_grid_thw=grid,
        frame_indices=indices,
        source_frame_count=np.asarray([metadata.total_num_frames], dtype=np.int64),
        source_fps=np.asarray([metadata.fps], dtype=np.float32),
    )
    with (output_dir / f"preprocessor_config_{model}.json").open("w") as file:
        json.dump(rust_processor_config(video_processor), file, indent=2)
        file.write("\n")
    print(f"{model}: frames={indices.tolist()} pixels={pixels.shape} grid={grid.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("tests/fixtures/video_golden"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "qwen_sampling.mp4"
    generate_source_video(video_path)
    for model in ("qwen2_vl", "qwen3_vl"):
        generate_model_golden(model, video_path, args.output_dir)


if __name__ == "__main__":
    main()
