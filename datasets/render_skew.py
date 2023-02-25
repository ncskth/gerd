from typing import Callable, Optional
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

def skew_tensor(image, shear_angle, shear):

    pad = int(image.size()[0]*2-image.size()[0])


    y_shear = shear*math.sin(math.pi*shear*shear_angle/180)
    x_shear = shear*math.cos(math.pi*shear*shear_angle/180)

    # define the transformation to be applied to images
    transform = torchvision.transforms.Compose([torchvision.transforms.Pad(pad),
        torchvision.transforms.RandomAffine(degrees=0, shear=[x_shear, x_shear+0.01, y_shear, y_shear+0.01], fill=0,
          interpolation=torchvision.transforms.InterpolationMode.NEAREST),
    ])

    skew_image = transform(torch.unsqueeze(image, dim=0)) 
    skew_image = skew_image[0]

    non_zero_indices = torch.nonzero(skew_image)

    x_min = torch.min(non_zero_indices[:,0])
    y_min = torch.min(non_zero_indices[:,1])
    x_max = torch.max(non_zero_indices[:,0])
    y_max = torch.max(non_zero_indices[:,1])

    return skew_image[x_min:x_max, y_min:y_max]


# function for original implementation
def render_shape(
    shape: Callable[[int, float, str], torch.Tensor],
    len: int,
    resolution: torch.Size,
    device: str,
    diameter: int = 80,
    shape_p: int = 1,
    bg_noise_p: float = 0.01,
    max_velocity: int = 1,
):
    """
    Draws a moving shape for `len` duration with maximum velocity of `max_velocity`.
    Arguments:
        shape (Callable): A function that generates a shape, expecting the shape size (int),
                          the bernouilli distribution probability (float), and the device
        len (int): The number of subsequent frames to generate
        resolution (torch.Size): The WxH resolution of the total frame. Should not be smaller than r
        device (str): The device on which the shape will be generated
        diameter (int): The diameter of the shape
        shape_p (float): The probability of drawing an event from a Bernouilli distribution
                         (1 = full shape, 0 = empty)
        bg_noise_p (float): The probability of drawing an event for added background noise from a
                         Bernouilli distribution (0 = no noise, 1 = full noise)
        max_velocity: The maximum allowed velocity between frames (1 = maximum moves one pixel)
    Returns:
        A tensor of size (len, resolution)
    """
    mask_r = diameter + 5
    images = torch.zeros(len, *resolution, dtype=torch.bool, device=device)
    labels = torch.zeros(len, 2)
    x = torch.randint(low=diameter, high=resolution[0] - diameter, size=(1,)).item()
    y = torch.randint(low=diameter, high=resolution[1] - diameter, size=(1,)).item()
    deltas = torch.distributions.Normal(loc=0, scale=0.2)
    velocity = torch.rand((2,), device=device)
    for i in range(images.shape[0]):
        x = x + velocity[0]
        y = y + velocity[1]
        if x < diameter or x > resolution[0] - diameter:
            velocity[0] = -1 * velocity[0]
        if y < diameter or y > resolution[1] - diameter:
            velocity[1] = -1 * velocity[1]
        velocity = (velocity + deltas.sample((2,)).to(device)).clip(
            -max_velocity, max_velocity
        )
        x = x.clip(mask_r, resolution[0] - mask_r - 1)
        y = y.clip(mask_r, resolution[1] - mask_r - 1)
        x_min = int(x.round() - diameter)
        x_max = x_min + diameter
        y_min = int(y.round() - diameter)
        y_max = y_min + diameter

        # Apply noise
        noise = (
            torch.distributions.Bernoulli(probs=bg_noise_p)
            .sample(images[i].shape)
            .to(device)
        )
        images[i] += noise.bool()

        # Fill in shape
        img = shape(diameter, p=shape_p, device=device)
        images[i, x_min:x_max, y_min:y_max] += img.bool()
        labels[i] = torch.tensor([(x_min + x_max) // 2, (y_min + y_max) // 2])

    return images.unsqueeze(1).float(), labels.unsqueeze(1)


# imrpoved shape rendering function
def render_shape_improved(
    shape_fn: Callable[[int, float, str], torch.Tensor],
    len: int,
    resolution: torch.Size,
    device: str,
    scale_change: bool,
    trans_change: bool,
    rotate_change: bool,
    skew_change: bool,
    rotate_angle: int = 0,
    diameter: Optional[int] = None,
    shape_p: float = 1,
    bg_noise_p: float = 0.01,
    max_velocity: int = 1,
    max_scale_change: int = 1,
    max_shear_change: float = 0.5,
):
    """
    Draws a moving shape for `len` duration with maximum velocity of `max_velocity`.
    Arguments:
        shape_fn (Callable[[int, float, str], torch.Tensor]): The function generating the shape
        len (int): The number of subsequent frames to generate
        resolution (torch.Size): The WxH resolution of the total frame. Should not be smaller than r
        device (str): The device on which the shape will be generated
        scale_change (Boolean): True if scale varies
        trans_change (Boolean): True if translation is present
        rotate_change (Boolean): True if shape rotates
        rotate_angle (int): The amount of rotation in degrees per frame. Defaults to 0
        diameter (Optional[int]): The diameter of the shape. Defaults to a random number between 50 and 200
        shape_p (float): The probability of drawing an event from a Bernouilli distribution
                         (1 = full shape, 0 = empty)
        bg_noise_p (float): The probability of drawing an event for added background noise from a
                         Bernouilli distribution (0 = no noise, 1 = full noise)
        max_velocity: The maximum allowed velocity between frames (1 = maximum moves one pixel)
        max_scale_change: The max change in the diameter of the shape between frames
    Returns:
        A tensor of size (len, resolution)
    """

    angle = 0
    shear = 0
    delta_shear = 0
    shear_velocity = 0

    mask_r = 5
    images = torch.zeros(len, *resolution, dtype=torch.bool, device=device)
    labels = torch.zeros(len, 2)

    # initialise starting x, y, diameter
    diameter = (
        diameter
        if diameter is not None
        else torch.randint(low=50, high=200, size=(1,)).item()
    )
    x = torch.randint(
        low=int(diameter / 2) + mask_r,
        high=resolution[0] - int(diameter / 2) - mask_r,
        size=(1,),
    ).item()
    y = torch.randint(
        low=int(diameter / 2) + mask_r,
        high=resolution[1] - int(diameter / 2) - mask_r,
        size=(1,),
    ).item()

    if skew_change:
        shear = torch.randint(low=0, high=40, size=(1,)).item()
        delta_shear = torch.distributions.Normal(loc=0, scale=0.2)
        shear_velocity =  torch.rand((1,), device=device)-0.5

      

    if scale_change:
        delta_scale = torch.distributions.Normal(loc=0, scale=0.2)
        scale = 2*torch.rand((1,), device=device)-1
    else:
        scale = torch.zeros((1,))

    if trans_change:
        deltas = torch.distributions.Normal(loc=0, scale=0.2)
        velocity = 2*torch.rand((2,), device=device)-1
    else:
        velocity = torch.zeros((2,), device=device)

    actual_diameter = diameter

    for i in range(images.shape[0]):

        x = x + velocity[0]
        y = y + velocity[1]


        noise = (
            torch.distributions.Bernoulli(probs=bg_noise_p)
            .sample(images[i].shape)
            .to(device)
        )
        images[i] += noise.bool()

        # Fill in shape
        img = shape_fn(int(diameter), p=shape_p, device=device)

        if skew_change:
            skewed_img = skew_tensor(img, angle, shear)
        else:
            skewed_img = img

        if rotate_change:
            rotated_img = rotate_tensor(skewed_img, angle)
        else:
            rotated_img = skewed_img


        angle = angle + rotate_angle

        if int(rotated_img.size()[0]) > min(
            resolution[0] - mask_r - 1, resolution[1] - mask_r - 1
        ):
            scale = -1 * abs(scale)
        if int(rotated_img.size()[0]) < min(resolution[0], resolution[1]) / 10:
            scale = abs(scale)
        if (
            x < int(rotated_img.size()[0] / 2) + mask_r
            or x > resolution[0] - int(rotated_img.size()[0] / 2) - mask_r
        ):
            velocity[0] = -1 * velocity[0]
            scale = -1 * abs(scale)
            rotated_img = rotate_tensor(rotated_img, rotate_angle)
        if (
            y < int(rotated_img.size()[1] / 2) + mask_r
            or y > resolution[1] - int(rotated_img.size()[1] / 2) - mask_r
        ):
            velocity[1] = -1 * velocity[1]
            scale = -1 * abs(scale)
            skewed_img = skew_tensor(rotated_img, angle, shear)
            rotated_img = rotate_tensor(skewed_img, rotate_angle)
        if (
            int(rotated_img.size()[0])
            > min(resolution[0] - mask_r - 10, resolution[1] - mask_r - 10)
            or x < int(rotated_img.size()[0] / 2) + mask_r
            or x > resolution[0] - int(rotated_img.size()[0] / 2) - mask_r
            or y < int(rotated_img.size()[1] / 2) + mask_r
            or y > resolution[1] - int(rotated_img.size()[1] / 2) - mask_r
        ):
            rotate_angle = -(rotate_angle)
            angle = angle + rotate_angle


            skewed_img = skew_tensor(img, angle, shear)

            rotated_img = rotate_tensor(skewed_img, angle)

        # brownian

        if scale_change:
            scale = (scale + delta_scale.sample((1,)).to(device)).clip(
                -max_scale_change, max_scale_change
            )

        if trans_change:
            velocity = (velocity + deltas.sample((2,)).to(device)).clip(
                -max_velocity, max_velocity
            )
        if skew_change:
            shear_velocity = (shear_velocity + delta_shear.sample((1,)).to(device)).clip(
                -max_shear_change, max_shear_change
            )


        actual_diameter = (diameter + scale[0]).item()
        diameter = int(round(actual_diameter))

        shear = (shear + shear_velocity[0]).clip(0, 40)


        x = x.clip(
            int(rotated_img.size()[0] / 2) + mask_r,
            resolution[0] - int(rotated_img.size()[0] / 2) - mask_r - 1,
        )
        y = y.clip(
            int(rotated_img.size()[1] / 2) + mask_r,
            resolution[1] - int(rotated_img.size()[1] / 2) - mask_r - 1,
        )

        x_min = int(x.round() - rotated_img.size()[0] / 2)
        x_max = int(x_min + rotated_img.size()[0])
        y_min = int(y.round() - rotated_img.size()[1] / 2)
        y_max = int(y_min + rotated_img.size()[1])


        images[i, x_min:x_max, y_min:y_max] += rotated_img.bool()
        labels[i] = torch.tensor([(x_min + x_max) // 2, (y_min + y_max) // 2])

    return images.unsqueeze(1).float(), labels.unsqueeze(1)


# if __name__ == "__main__":
#     #from shapes import circle_improved

#     render_shape_improved(
#         square_improved,
#         len=128,
#         resolution=(256, 256),
#         shape_p=0.8,
#         bg_noise_p=0.01,
#         device="cpu",
#         rotate_angle=1,
#         scale_change=False,
#         trans_change=False,
#         rotate_change=False,
#         skew_change=True,
#     )