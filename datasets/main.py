import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import signal
import traceback
from typing import List, NamedTuple, Optional

import torch
import tqdm.asyncio
from render import render_shape
from shapes import *


class DatasetParameters(NamedTuple):
    resolution: torch.Size
    event_density: float
    bg_density: float
    shape_density: float = 1
    polarity: bool = True

    upsampling_factor: int = 8
    upsampling_cutoff: Optional[float] = None
    bg_files: Optional[List[str]] = None
    device: str = "cuda"
    length: int = 128

    transformation: bool = False
    scale: bool = False
    rotate: bool = False
    shear: bool = False
    max_velocity: float = 0.2


def superimpose_data(file, images, p: DatasetParameters):
    _, _, frames, _, _, _, _ = torch.load(file, map_location=p.device)
    # Reduce polarity
    if not p.polarity:
        frames = frames.sum(-1, keepdim=True)
    # Crop
    frames = frames[:, : p.resolution[0], : p.resolution[1]]
    # Permute to TCHW
    frames = frames.permute(0, 3, 2, 1)
    # Normalize
    return (images + frames).clip(0, 1)


def render_shapes(p: DatasetParameters):
    shapes = []
    labels = []
    for fn in [circle, square, triangle]:
        s, l = render_shape(
            fn,
            len=p.length,
            resolution=p.resolution,
            shape_p=p.shape_density,
            bg_noise_p=p.bg_density,
            event_p=p.event_density,
            device=p.device,
            scale_change=p.scale,
            trans_change=p.transformation,
            rotate_change=p.rotate,
            shear_change=p.shear,
            upscale_factor=p.upsampling_factor,
            upscale_cutoff=p.upsampling_cutoff,
            max_trans_change=p.max_velocity,
            max_scale_change=p.max_velocity,
            max_angle_change=p.max_velocity,
            max_shear_change=p.max_velocity,
        )
        shapes.append(s)
        labels.append(l)

    images = torch.stack(shapes).sum(0)
    if not p.polarity:
        images = images.sum(1, keepdim=True)
    images = images.clip(0, 1)
    labels = torch.stack(labels).permute(1, 0, 2, 3)
    return images, labels


async def render_points(output_folder, index, p: DatasetParameters):
    filename = output_folder / f"{index}.dat"
    try:
        with torch.inference_mode():
            images, labels = render_shapes(p)
            if p.bg_files is not None:
                images = superimpose_data(
                    p.bg_files[index % len(p.bg_files)], images, p
                )

            t = [images.clip(0, 1).to_sparse(), labels]
            torch.save(t, filename)
    except Exception as e:
        print(e)
        traceback.print_exc()


async def main(args):

    if args.seed is not None:
        torch.manual_seed(args.seed)

    n = torch.arange(args.n)
    threads = torch.cuda.device_count()
    resolution = (300, 300)
    root_folder = Path(args.root)
    if not root_folder.exists():
        root_folder.mkdir()

    bg_files = None
    if args.root_bg is not None:
        bg_folder = Path(args.root_bg)
        if bg_folder.exists():
            bg_files = list(bg_folder.glob("*.dat"))
            sorted(bg_files)

    # Permutations of transformations
    transformation_combinations = [
        torch.tensor(
            [args.translation, args.scaling, args.rotation, args.shearing]
        ).int()
    ]

    # Start multiprocessing
    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGINT, sigint_handler)
    futures = []
    for event_p in args.event_densities:
        for max_velocity in args.max_velocities:
            for comb in transformation_combinations:
                combination_name = (
                    str(comb.tolist()).replace(", ", "").replace("True", "1")[1:-1]
                )
                output_folder = (
                    root_folder
                    / f"v{max_velocity:.2f}-p{event_p:.2f}-{combination_name}"
                )
                for i in n:

                    parameters = DatasetParameters(
                        resolution=resolution,
                        bg_density=0.001,
                        bg_files=bg_files,
                        event_density=event_p,
                        polarity=args.polarity,
                        device=f"cuda:{i % threads}",
                        transformation=comb[0],
                        scale=comb[1],
                        rotate=comb[2],
                        shear=comb[3],
                        max_velocity=max_velocity,
                    )
                    if not output_folder.exists():
                        output_folder.mkdir()
                    f = render_points(output_folder, i, parameters)
                    futures.append(f)
    for f in tqdm.asyncio.tqdm.as_completed(futures):
        await f


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Render dataset")
    parser.add_argument("n", type=int, help="Number of samples per event density")
    parser.add_argument("root", type=str, help="Path to output directory")
    parser.add_argument(
        "--root_bg",
        type=str,
        default=None,
        help="Location of dataset to use as background",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to initialize random dataset mapping",
    )
    parser.add_argument(
        "--translation",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--scaling",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--rotation",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--shearing",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--polarity",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--event_densities",
        nargs="+",
        type=float,
        default=[0.8],
        help="Event density as a list of floats",
    )
    parser.add_argument(
        "--max_velocities",
        type=float,
        nargs="+",
        default=[0.2],
        help="Max velocities as a list of float (1 = 1px change/frame)",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
