#!/usr/bin/env python3
"""Convert a GERD .dat recording to an MP4 video.

Usage:
    python scripts/to_mp4.py recording.dat
    python scripts/to_mp4.py recording.dat -o output.mp4 --fps 60
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def frames_to_rgb(frames: torch.Tensor) -> np.ndarray:
    """Convert event frames [T, 2, H, W] to RGB uint8 [T, H, W, 3].

    Color scheme: ON events → green, OFF events → red, background → white.
    """
    on = frames[:, 0]   # [T, H, W]
    off = frames[:, 1]  # [T, H, W]

    r = (1.0 - off).clamp(0, 1)
    g = (1.0 - on).clamp(0, 1)
    b = (1.0 - on - off).clamp(0, 1)

    rgb = torch.stack([r, g, b], dim=-1)  # [T, H, W, 3]
    return (rgb * 255).byte().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        prog="to_mp4",
        description="Convert a GERD .dat recording to an MP4 video.",
    )
    parser.add_argument("input", help="Input .dat file")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output .mp4 file (default: input stem + .mp4)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".mp4")

    print(f"Loading {input_path} ...")
    sparse_frames, labels = torch.load(input_path, map_location=args.device)
    frames = sparse_frames.to_dense()  # [T, C, H, W]

    if frames.shape[1] == 1:
        # No-polarity recording: treat single channel as ON, zero OFF
        frames = torch.cat([frames, torch.zeros_like(frames)], dim=1)

    print(f"  {frames.shape[0]} frames, resolution {frames.shape[3]}x{frames.shape[2]}")
    rgb = frames_to_rgb(frames)  # [T, H, W, 3] uint8

    try:
        import imageio
    except ImportError:
        print("imageio is required. Install with: pip install imageio[ffmpeg]", file=sys.stderr)
        sys.exit(1)

    print(f"Writing {output_path} at {args.fps} fps ...")
    imageio.mimwrite(str(output_path), rgb, fps=args.fps, codec="libx264")
    print("Done.")


if __name__ == "__main__":
    main()
