from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
import math
import torch
import torchvision


def events_to_frames(frames, polarity: bool = False):
    if len(frames.shape) == 3:
        frames = frames.unsqueeze(-1).repeat(1, 1, 1, 3)
    else:
        if not polarity:
            frames = frames.abs().sum(-1)
        elif polarity:
            frames = torch.concat(
                [frames, torch.zeros(*frames.shape[:-1], 1, device=frames.device)],
                dim=-1,
            )
    frames = ((frames / frames.max()) * 255).int()
    return frames


def rotate_tensor(input, x):
    rotated_input = torchvision.transforms.functional.rotate(
        torch.unsqueeze(input, dim=0),
        x,
        expand=True,
        fill=0,
        interpolation=torchvision.transforms.InterpolationMode.NEAREST,
    )
    return rotated_input[0]


def shear_tensor(image, shear_angle, shear):
    if shear == 0:
        return image

    pad = int(image.size()[0] * 2)

    y_shear = shear * math.sin(math.pi * shear * shear_angle / 180)
    x_shear = shear * math.cos(math.pi * shear * shear_angle / 180)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Pad(pad),
            torchvision.transforms.RandomAffine(
                degrees=0,
                shear=[x_shear, x_shear + 0.01, y_shear, y_shear + 0.01],
                fill=0,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            ),
        ]
    )

    shear_image = transform(torch.unsqueeze(image, dim=0))
    shear_image = shear_image[0]

    non_zero_indices = torch.nonzero(shear_image)
    if non_zero_indices.shape[0] == 0:
        return image

    x_min = torch.min(non_zero_indices[:, 0])
    y_min = torch.min(non_zero_indices[:, 1])
    x_max = torch.max(non_zero_indices[:, 0])
    y_max = torch.max(non_zero_indices[:, 1])

    return shear_image[x_min:x_max, y_min:y_max]


ZERO_DISTRIBUTION = torch.distributions.Categorical(torch.tensor([1]))


def blit_shape(shape, bg, x, y, device):
    width = shape.shape[0]
    height = shape.shape[1]
    offset_x = x - torch.round(x)
    offset_y = y - torch.round(y)
    x_lin = torch.linspace(-1, 1, width).to(device) - offset_x / (width / 2)
    y_lin = torch.linspace(-1, 1, height).to(device) - offset_y / (height / 2)
    coo = (
        torch.stack(torch.meshgrid(x_lin, y_lin, indexing="xy"), -1)
        .unsqueeze(0)
        .float()
    )
    sampled_img = (
        torch.nn.functional.grid_sample(
            shape.unsqueeze(0).unsqueeze(0).float(),
            coo,
            align_corners=False,
            padding_mode="zeros",
        )
        .squeeze()
        .T
    )
    f = lambda x: round(x.item())
    left_x = max(0, f(x))
    right_x = min(left_x + width, bg.shape[0])
    left_y = max(0, f(y))
    right_y = min(left_y + height, bg.shape[1])
    bg[left_x:right_x, left_y:right_y] = sampled_img[
        : right_x - left_x, : right_y - left_y
    ]
    return offset_x, offset_y, left_x, left_y, sampled_img


@dataclass
class RenderParameters:
    """
    Parameters for rendering a shape.

    The arguments are split into rendering parameters and transformation parameters.
    Rendering parameters configure the simulation (resolution, frame count, event density, etc.).
    Transformation parameters follow a consistent pattern per transformation:

      <transform>_start    : initial state. "uniform" = sample randomly from the valid range,
                             float (or tuple for translate) = fixed value, None = not applied.
                             translate_start cannot be None; the shape always has a position.
      <transform>_velocity : constant rate of change per frame. 0.0 (default) = static.
                             Units: translate = actual pixels/frame, scale = actual pixels/frame,
                             rotate = degrees/frame, shear = shear units/frame.
      <transform>_velocity_delta : optional callable(n) -> Tensor adding acceleration each step.
                                   None (default) = constant velocity.
      <transform>_velocity_max   : optional clip bound applied after delta. None = no clipping.
                                   Same units as the corresponding velocity.

    Example:
        Translate a square at 1 pixel/frame for 10 frames from a random start:
        >>> p = RenderParameters(
        ...     resolution=torch.Size([256, 256]),
        ...     length=10,
        ...     translate_start="uniform",
        ...     translate_velocity=(1.0, 0.0))
        >>> render_shape(shapes.square, p)

    Arguments:
        resolution (torch.Size): WxH resolution of the output frame
        length (int): Number of frames to generate. Default = 128
        event_density (float): Probability of sampling from the frame diff. Default = 1
        shape_density (float): Probability of sampling the shape contour. Default = 1
        bg_noise_density (float): Background noise probability. Default = 0.001
        polarity (bool): Output with polarity (PxXxY) if True, single channel otherwise. Default = True
        warmup_steps (int): Warmup steps to initialise the integrator. Default = 5
        min_fraction (float): Minimum shape scale relative to the viewport. Default = 0.05
        max_fraction (float): Maximum shape scale relative to the viewport. Default = 0.7
        border_radius (int): Border margin around the shape. Default = 5
        initial_integration_distribution: Distribution to initialise the IAF state.
            Defaults to Uniform(-0.9 * cutoff, 0.9 * cutoff).
        upsampling_factor (int): Internal upsampling factor for sub-pixel motion. Default = 8
        upsampling_cutoff (float): IAF threshold. Default = 0.5
        device (str): Torch device. Default = "cuda"
    """

    resolution: torch.Size
    length: int = 128
    event_density: float = 1
    shape_density: float = 1
    bg_noise_density: float = 0.001
    polarity: bool = True
    warmup_steps: int = 5
    min_fraction: float = 0.05
    max_fraction: float = 0.7
    border_radius: int = 5
    initial_integration_distribution: Optional[torch.distributions.Distribution] = None

    upsampling_factor: int = 8
    upsampling_cutoff: float = 1 / 2
    device: str = "cuda"

    # Global velocity bound, used as fallback when a transformation's _velocity_max is not set.
    # Only applied when a _velocity_delta (acceleration) is active.
    transformation_velocity_max: Optional[float] = None

    # Translation: shape always has a position
    translate_start: Union[str, Tuple[float, float]] = "uniform"
    translate_velocity: Union[float, Tuple[float, float]] = 0.0
    translate_velocity_delta: Optional[Callable[[int], torch.Tensor]] = None
    translate_velocity_max: Optional[float] = None

    # Scale: None = not applied (default size)
    scale_start: Union[str, float, None] = None
    scale_velocity: float = 0.0
    scale_velocity_delta: Optional[Callable[[int], torch.Tensor]] = None
    scale_velocity_max: Optional[float] = None

    # Rotation: None = not applied (angle fixed at 0)
    rotate_start: Union[str, float, None] = None
    rotate_velocity: float = 0.0
    rotate_velocity_delta: Optional[Callable[[int], torch.Tensor]] = None
    rotate_velocity_max: Optional[float] = None

    # Shear: None = not applied
    shear_start: Union[str, float, None] = None
    shear_max: float = 30
    shear_velocity: float = 0.0
    shear_velocity_delta: Optional[Callable[[int], torch.Tensor]] = None
    shear_velocity_max: Optional[float] = None


class IAFSubtractReset(torch.nn.Module):

    def __init__(self, cutoff: float, distribution: torch.distributions.Distribution):
        super().__init__()
        self.cutoff = cutoff
        self.distribution = distribution

    def forward(self, x, state=None):
        if state is None:
            state = self.distribution.sample(x.shape).to(x.device)
        v_new = state + x
        z_pos = v_new > self.cutoff
        z_neg = v_new < -self.cutoff
        v_new = v_new - z_pos * self.cutoff + z_neg * self.cutoff
        return torch.stack([z_pos, z_neg]), v_new


def render_shape(
    shape_fn: Callable[[int, float, str], torch.Tensor],
    p: RenderParameters,
):
    """
    Draws a moving shape for `length` timesteps.
    Arguments:
        shape_fn: function(size, p, device) -> Tensor generating the shape
        p: RenderParameters controlling the simulation
    Returns:
        Tuple of (events tensor of shape (length, 2, *resolution), labels tensor)
    """

    bg_noise_dist = (
        torch.distributions.Bernoulli(probs=p.bg_noise_density / 2)
        if p.polarity
        else p.bg_noise_density
    )
    shape_density = (
        torch.pow(torch.as_tensor(p.shape_density), 0.25)
        if p.polarity
        else p.shape_density
    )
    event_dist = torch.distributions.Bernoulli(probs=p.event_density)

    mask_r = p.border_radius
    images = torch.zeros(p.length, 2, *p.resolution, dtype=torch.bool, device=p.device)
    labels = torch.zeros(p.length, 2)

    if p.initial_integration_distribution is None:
        initial_distribution = torch.distributions.Uniform(
            -p.upsampling_cutoff * 0.9, p.upsampling_cutoff * 0.9
        )
    else:
        initial_distribution = p.initial_integration_distribution

    neuron_population = IAFSubtractReset(p.upsampling_cutoff, initial_distribution)
    neuron_state = None
    current_image = None
    previous_image = None

    min_resolution = torch.as_tensor(min(p.resolution[0], p.resolution[1]))
    min_size = float((p.min_fraction * min_resolution).int())
    max_size = float((p.max_fraction * min_resolution).int())
    resolution_upscaled = torch.as_tensor(p.resolution) * p.upsampling_factor

    # --- Initialise scale ---
    if p.scale_start is None:
        scale = float(min_resolution // 3)
    elif p.scale_start == "uniform":
        scale = float(torch.rand(1).item() * (max_size - min_size) + min_size)
    else:
        scale = float(p.scale_start)
    scale_velocity = float(p.scale_velocity)

    # --- Initialise position (stored in upsampled coordinates) ---
    scaled_buffer = (mask_r + scale / 2) * p.upsampling_factor
    if p.translate_start == "uniform":
        x = torch.empty(1, device=p.device).uniform_(
            scaled_buffer, float(resolution_upscaled[0]) - scaled_buffer
        )
        y = torch.empty(1, device=p.device).uniform_(
            scaled_buffer, float(resolution_upscaled[1]) - scaled_buffer
        )
    else:
        x_s, y_s = p.translate_start
        x = torch.tensor([float(x_s) * p.upsampling_factor], device=p.device)
        y = torch.tensor([float(y_s) * p.upsampling_factor], device=p.device)

    # Translate velocity: convert from actual px/frame to upsampled px/frame
    if isinstance(p.translate_velocity, (int, float)):
        trans_velocity = (
            torch.tensor([float(p.translate_velocity)] * 2, device=p.device)
            * p.upsampling_factor
        )
    elif torch.is_tensor(p.translate_velocity):
        trans_velocity = p.translate_velocity.float().to(p.device) * p.upsampling_factor
    else:
        vx, vy = p.translate_velocity
        trans_velocity = (
            torch.tensor([float(vx), float(vy)], device=p.device) * p.upsampling_factor
        )

    # --- Initialise rotation ---
    if p.rotate_start is None:
        angle = 0.0
    elif p.rotate_start == "uniform":
        angle = float(torch.randint(0, 360, (1,)).item())
    else:
        angle = float(p.rotate_start)
    rotate_velocity = float(p.rotate_velocity)

    # --- Initialise shear ---
    if p.shear_start is None:
        shear = 0.0
    elif p.shear_start == "uniform":
        shear = float(torch.randint(0, int(p.shear_max), (1,)).item())
    else:
        shear = float(p.shear_start)
    shear_velocity = float(p.shear_velocity)

    # --- Main loop ---
    for i in range(-p.warmup_steps - 1, images.shape[0]):
        # Generate shape at current scale
        img = shape_fn(
            int(scale * p.upsampling_factor), p=shape_density, device=p.device
        )

        # Translate
        x = x + trans_velocity[0]
        y = y + trans_velocity[1]
        x = x.clip(scaled_buffer, resolution_upscaled[0] - scaled_buffer)
        y = y.clip(scaled_buffer, resolution_upscaled[1] - scaled_buffer)
        if p.translate_velocity_delta is not None:
            delta = p.translate_velocity_delta(2).to(p.device) * p.upsampling_factor
            trans_velocity = trans_velocity + delta
            effective_max = p.translate_velocity_max or p.transformation_velocity_max
            if effective_max is not None:
                trans_velocity = trans_velocity.clip(
                    -effective_max * p.upsampling_factor,
                    effective_max * p.upsampling_factor,
                )

        # Rotate
        if p.rotate_start is not None:
            angle = angle + rotate_velocity
            if p.rotate_velocity_delta is not None:
                rotate_velocity += float(p.rotate_velocity_delta(1).to(p.device))
                effective_max = p.rotate_velocity_max or p.transformation_velocity_max
                if effective_max is not None:
                    rotate_velocity = max(-effective_max, min(effective_max, rotate_velocity))
        img = rotate_tensor(img, float(angle))

        # Shear
        if p.shear_start is not None:
            if shear >= p.shear_max or shear <= -p.shear_max:
                shear_velocity = -shear_velocity
            if p.shear_velocity_delta is not None:
                shear_velocity += float(p.shear_velocity_delta(1).to(p.device))
                effective_max = p.shear_velocity_max or p.transformation_velocity_max
                if effective_max is not None:
                    shear_velocity = max(-effective_max, min(effective_max, shear_velocity))
            shear = float(min(p.shear_max, max(-p.shear_max, shear + shear_velocity)))
        img = shear_tensor(img, 0, shear)

        # Scale: bounce at size bounds
        if scale >= max_size:
            scale = max_size
            scale_velocity = -scale_velocity
        elif scale <= min_size:
            scale = min_size
            scale_velocity = -scale_velocity

        # Translation: bounce at frame bounds
        if x <= scaled_buffer or x >= resolution_upscaled[0] - scaled_buffer:
            trans_velocity[0] *= -1
        if y <= scaled_buffer or y >= resolution_upscaled[1] - scaled_buffer:
            trans_velocity[1] *= -1

        # Scale update
        if p.scale_start is not None:
            if p.scale_velocity_delta is not None:
                scale_velocity += float(p.scale_velocity_delta(1).to(p.device))
                effective_max = p.scale_velocity_max or p.transformation_velocity_max
                if effective_max is not None:
                    scale_velocity = max(-effective_max, min(effective_max, scale_velocity))
            scale = scale + scale_velocity
            scaled_buffer = (mask_r + scale / 2) * p.upsampling_factor

        # Blit shape onto upsampled frame
        x_center, y_center = torch.tensor([x, y]) - torch.tensor(img.shape) / 2
        current_image = torch.zeros(*resolution_upscaled, device=p.device)
        blit_shape(img, current_image, x_center, y_center, p.device)

        # Downsample and compute events
        if previous_image is not None:
            downsample = lambda x: torch.nn.functional.interpolate(
                x.unsqueeze(0).unsqueeze(0),
                scale_factor=1 / p.upsampling_factor,
                mode="bilinear",
                antialias=False,
            )
            prev_down = downsample(previous_image)
            curr_down = downsample(current_image)
            downsampled_diff = (prev_down - curr_down).squeeze()
            noise_mask = event_dist.sample((downsampled_diff.shape)).bool().to(p.device)
            ch1, neuron_state = neuron_population(downsampled_diff, neuron_state)

            if i >= 0:
                images[i] = ch1.bool() & noise_mask
                labels[i] = torch.tensor([x, y]) / p.upsampling_factor

        previous_image = current_image

        if p.device.startswith("cuda"):
            torch.cuda.empty_cache()

    images += bg_noise_dist.sample(images.shape).to(p.device).bool()
    return images.float().clip(0, 1), labels


if __name__ == "__main__":
    import shapes

    p = RenderParameters(
        torch.Size((300, 300)),
        20,
        scale_start="uniform",
        scale_velocity=1.0,
        rotate_start=10.0,
        upsampling_factor=4,
        upsampling_cutoff=1 / 4,
        bg_noise_density=0,
    )
    render_shape(shapes.triangle, p)
