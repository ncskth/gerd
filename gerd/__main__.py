import argparse
from pathlib import Path
import signal
import traceback
from typing import List, NamedTuple, Optional, Tuple, Union

import ray
import torch
import tqdm
import yaml

from gerd.render import render_shape, RenderParameters
from gerd.shapes import circle, square, triangle


class DatasetParameters(NamedTuple):
    resolution: torch.Size
    event_density: float
    bg_density: float
    shape_density: float = 1
    polarity: bool = True

    upsampling_factor: int = 8
    upsampling_cutoff: float = 1 / 2
    bg_files: Optional[List[str]] = None
    device: str = "cuda"
    length: int = 128

    # For each transformation, _start sets the initial state and _velocity sets the rate of change.
    # _start: "uniform" = sample randomly from valid range, float = fixed value, None = not applied
    # _velocity: 0.0 = static, "uniform" = sample randomly from [-max_velocity, max_velocity], float = fixed rate
    # Note: translate_start=None is not supported; the shape always has a position.
    translate_start: Union[str, Tuple[float, float]] = "uniform"
    translate_velocity: Union[str, float, Tuple[float, float]] = 0.0

    scale_start: Union[str, float, None] = None
    scale_velocity: Union[str, float] = 0.0

    rotate_start: Union[str, float, None] = None
    rotate_velocity: Union[str, float] = 0.0

    shear_start: Union[str, float, None] = None
    shear_velocity: Union[str, float] = 0.0

    max_velocity: float = 0.2


def _transform_kwargs(section: Optional[dict], default_start) -> Tuple:
    """Extract (start, velocity) from a YAML transformation section, converting lists to tuples."""
    if section is None:
        section = {}
    start = section.get("start", default_start)
    velocity = section.get("velocity", 0.0)
    if isinstance(start, list):
        start = tuple(start)
    if isinstance(velocity, list):
        velocity = tuple(velocity)
    return start, velocity


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_to_dataset_params(
    config: dict,
    event_density: float,
    max_velocity: float,
    bg_files=None,
) -> DatasetParameters:
    translate_start, translate_velocity = _transform_kwargs(
        config.get("translate"), "uniform"
    )
    scale_start, scale_velocity = _transform_kwargs(config.get("scale"), None)
    rotate_start, rotate_velocity = _transform_kwargs(config.get("rotate"), None)
    shear_start, shear_velocity = _transform_kwargs(config.get("shear"), None)

    return DatasetParameters(
        resolution=tuple(config.get("resolution", [300, 300])),
        length=config.get("length", 128),
        event_density=event_density,
        bg_density=config.get("bg_density", 0.001),
        shape_density=config.get("shape_density", 1.0),
        polarity=config.get("polarity", True),
        device=config.get("device", "cuda"),
        upsampling_factor=config.get("upsampling_factor", 8),
        upsampling_cutoff=config.get("upsampling_cutoff", 0.5),
        bg_files=bg_files,
        max_velocity=max_velocity,
        translate_start=translate_start,
        translate_velocity=translate_velocity,
        scale_start=scale_start,
        scale_velocity=scale_velocity,
        rotate_start=rotate_start,
        rotate_velocity=rotate_velocity,
        shear_start=shear_start,
        shear_velocity=shear_velocity,
    )


def superimpose_data(file, images, p: DatasetParameters):
    _, _, frames, _, _, _, _ = torch.load(file, map_location=p.device)
    if not p.polarity:
        frames = frames.sum(-1, keepdim=True)
    frames = frames[:, : p.resolution[0], : p.resolution[1]]
    frames = frames.permute(0, 3, 2, 1)
    return (images + frames).clip(0, 1)


def _sample_uniform(cat, device, n, max_velocity):
    """Sample n values uniformly from {-max_velocity, +max_velocity}."""
    return (cat.sample((n,)).to(device) - 0.5) * 2 * max_velocity


def render_shapes(p: DatasetParameters):
    shapes = []
    labels = []
    for fn in [circle, square, triangle]:
        cat = torch.distributions.Categorical(torch.tensor([0.5, 0.5]))
        args = {}

        # --- Translate ---
        if isinstance(p.translate_start, tuple):
            args["translate_start"] = p.translate_start
        # else "uniform": default in RenderParameters

        if p.translate_velocity == "uniform":
            args["translate_velocity"] = tuple(
                _sample_uniform(cat, p.device, 2, p.max_velocity).tolist()
            )
        elif p.translate_velocity != 0.0:
            args["translate_velocity"] = p.translate_velocity

        # --- Scale ---
        if p.scale_start is not None:
            args["scale_start"] = p.scale_start  # "uniform" or float
            if p.scale_velocity == "uniform":
                args["scale_velocity"] = float(
                    _sample_uniform(cat, p.device, 1, p.max_velocity)
                )
            elif p.scale_velocity != 0.0:
                args["scale_velocity"] = float(p.scale_velocity)

        # --- Rotate ---
        if p.rotate_start is not None:
            args["rotate_start"] = p.rotate_start  # "uniform" or float
            if p.rotate_velocity == "uniform":
                args["rotate_velocity"] = float(
                    _sample_uniform(cat, p.device, 1, p.max_velocity)
                )
            elif p.rotate_velocity != 0.0:
                args["rotate_velocity"] = float(p.rotate_velocity)

        # --- Shear ---
        if p.shear_start is not None:
            args["shear_start"] = p.shear_start  # "uniform" or float
            if p.shear_velocity == "uniform":
                args["shear_velocity"] = float(
                    _sample_uniform(cat, p.device, 1, p.max_velocity)
                )
            elif p.shear_velocity != 0.0:
                args["shear_velocity"] = float(p.shear_velocity)

        render_p = RenderParameters(
            length=p.length,
            resolution=p.resolution,
            shape_density=p.shape_density,
            bg_noise_density=p.bg_density,
            event_density=p.event_density,
            device=p.device,
            upsampling_factor=p.upsampling_factor,
            upsampling_cutoff=p.upsampling_cutoff,
            transformation_velocity_max=p.max_velocity,
            **args,
        )
        s, l = render_shape(fn, render_p)
        shapes.append(s)
        labels.append(l)

    images = torch.stack(shapes).sum(0)
    if not p.polarity:
        images = images.sum(1, keepdim=True)
    images = images.clip(0, 1)
    labels = torch.stack(labels).permute(1, 0, 2)
    return images, labels


@ray.remote(num_gpus=1)
def render_points(output_folder, index, p: DatasetParameters):
    filename = output_folder / f"{index}.dat"
    try:
        with torch.inference_mode():
            images, labels = render_shapes(p)
            if p.bg_files is not None:
                images = superimpose_data(
                    p.bg_files[index % len(p.bg_files)], images, p
                )
            torch.save([images.clip(0, 1).to_sparse(), labels], filename)
    except Exception as e:
        print(e)
        traceback.print_exc()


def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)

    config = load_config(args.config)
    config_name = Path(args.config).stem

    root_folder = Path(args.root)
    root_folder.mkdir(exist_ok=True)

    bg_files = None
    if args.root_bg is not None:
        bg_folder = Path(args.root_bg)
        if bg_folder.exists():
            bg_files = sorted(bg_folder.glob("*.dat"))

    event_densities = config.get("event_densities", [1.0])
    max_velocities = config.get("max_velocities", [0.2])

    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGINT, sigint_handler)

    futures = []
    for event_density in event_densities:
        for max_velocity in max_velocities:
            output_folder = (
                root_folder / f"{config_name}-v{max_velocity:.2f}-p{event_density:.2f}"
            )
            output_folder.mkdir(exist_ok=True)

            parameters = config_to_dataset_params(
                config, event_density, max_velocity, bg_files
            )
            for i in range(args.n):
                futures.append(render_points.remote(output_folder, i, parameters))

    remaining = futures
    with tqdm.tqdm(total=len(futures)) as pbar:
        while remaining:
            done, remaining = ray.wait(remaining, num_returns=1)
            pbar.update(len(done))


def cli():
    parser = argparse.ArgumentParser(
        prog="gerd",
        description=(
            "Generate synthetic event-camera datasets for object tracking tasks.\n"
            "Shapes (circle, square, triangle) are rendered subject to configurable\n"
            "affine transformations (translation, scaling, rotation, shearing).\n"
            "Output is saved as sparse PyTorch tensors, one file per sample."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  gerd 1000 /data example.yaml          # generate 1000 samples\n"
            "  gerd 10 /tmp/test example.yaml --seed 42  # quick reproducible test"
        ),
    )
    parser.add_argument("n", type=int, help="Number of samples per configuration")
    parser.add_argument("root", type=str, help="Output directory")
    parser.add_argument("config", type=str, help="Path to YAML configuration file (see example.yaml)")
    parser.add_argument(
        "--root_bg", type=str, default=None,
        help="Directory of background .dat files to superimpose on generated frames"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()
    ray.init()
    main(args)


if __name__ == "__main__":
    cli()
