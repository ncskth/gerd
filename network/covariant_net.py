import torch
import torch.nn as nn


def generate_time_constants(num_tc, middle_tc, ratio):
    """
    Generate log-spaced time constants centered around middle_tc.
    """
    middle = num_tc // 2
    indices = torch.arange(num_tc, device=middle_tc.device, dtype=middle_tc.dtype)
    exponents = indices - middle
    taus = middle_tc * (ratio ** exponents)
    return taus


class LeakyIntegrator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, prev_state, tau):
        if prev_state is None:
            prev_state = torch.zeros_like(x)

        alpha = torch.exp(-1.0 / tau)
        new_state = alpha * prev_state + (1 - alpha) * x
        return new_state


class SpatioTemporalConv2d(nn.Module):
    def __init__(
        self,
        conv: nn.Conv2d,
        time_constants_num: int,
        middle_TC: float = 10.0,
        TC_ratio: float = 2.0,
        expand: bool = False,
    ):
        super().__init__()

        self.time_constants_num = time_constants_num
        self.TC_ratio = TC_ratio
        self.expand = expand
        self.middle_TC = nn.Parameter(torch.tensor(float(middle_TC)))
        self.conv = conv
        self.li = LeakyIntegrator()
        self.states = [None] * time_constants_num


    def forward(self, input):
        """
        input:
            expand=True  -> (T, B, C, H, W)
            expand=False -> (T, B, C*num_TC, H, W)

        output:
            (T, B, C*num_TC, H, W)
        """
        taus = generate_time_constants(
            self.time_constants_num,
            self.middle_TC,
            self.TC_ratio
        )
        T = input.shape[0]
        outputs = []

        for t in range(T):
            x = input[t]  # (B, C, H, W) or (B, C*num_TC, H, W)

            if self.expand:
                # --- Single conv ---
                y = self.conv(x)  # (B, C, H, W)

                tc_outputs = []
                for i in range(self.time_constants_num):
                    self.states[i] = self.li(y, self.states[i], taus[i])
                    tc_outputs.append(self.states[i])

                out_t = torch.cat(tc_outputs, dim=1)  # (B, C*num_TC, H, W)

            else:
                # --- Already expanded channels ---
                B, C_total, H, W = x.shape
                C = C_total // self.time_constants_num

                x_split = x.view(B, self.time_constants_num, C, H, W)

                tc_outputs = []

                for i in range(self.time_constants_num):
                    xi = x_split[:, i]  # (B, C, H, W)

                    # shared conv applied per group
                    yi = self.conv(xi)

                    self.states[i] = self.li(yi, self.states[i], taus[i])
                    tc_outputs.append(self.states[i])

                out_t = torch.cat(tc_outputs, dim=1)  # (B, C*num_TC, H, W)

            outputs.append(out_t)

        return torch.stack(outputs, dim=0)

    def reset_state(self):
        self.states = [None] * self.time_constants_num



class TCHead(nn.Module):
    def __init__(self, in_channels, num_TC):
        super().__init__()
        self.num_TC = num_TC
        self.in_channels = in_channels  # this is C

        # Conv to produce 3 heatmaps
        self.head = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)

    def forward(self, x, epoch=None):
        """
        x: (T, B, C*num_TC, H, W)
        returns:
            heatmaps: (T, B, 3, H, W)
            coords:   (T, B, 3, 2)
        """

        T, B, C_total, H, W = x.shape
        C = C_total // self.num_TC

        # --- reshape into TC groups ---
        x = x.view(T, B, self.num_TC, C, H, W)
        x = x.mean(dim=2)  # (T, B, C, H, W)

        # --- apply conv head ---
        x_reshaped = x.view(T * B, C, H, W)
        heatmaps = self.head(x_reshaped)  # (T*B, 3, H, W)
        heatmaps = heatmaps.view(T, B, 3, H, W)

        # --- get coordinates via soft-argmax ---
        coords = self.soft_argmax_2d(heatmaps)

        return heatmaps, coords

    def soft_argmax_2d(self, heatmaps):
        """
        heatmaps: (T, B, 3, H, W)
        returns:  (T, B, 3, 2)  -> (y, x)
        """
        T, B, C, H, W = heatmaps.shape
        # Flatten spatial dimensions
        heatmaps_flat = heatmaps.view(T, B, C, H * W)

        # Softmax over spatial locations
        probs = torch.softmax(heatmaps_flat, dim=-1)  # (T, B, C, H*W)

        # Create coordinate grids
        xs = torch.linspace(0, W - 1, W, device=heatmaps.device)
        ys = torch.linspace(0, H - 1, H, device=heatmaps.device)

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)

        # Flatten grids
        grid_x = grid_x.reshape(-1)  # (H*W,)
        grid_y = grid_y.reshape(-1)  # (H*W,)

        # Compute expected coordinates
        x_coord = (probs * grid_x).sum(dim=-1)  # (T, B, C)
        y_coord = (probs * grid_y).sum(dim=-1)  # (T, B, C)

        # Stack into final coordinates: (T, B, C, 2)
        coords = torch.stack([x_coord, y_coord], dim=-1)

        return coords    

class SPT_Net(nn.Module):
    def __init__(self, warmup = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kernel_size = 11
        self.spt_conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.temp_cov1 = SpatioTemporalConv2d(self.spt_conv1, time_constants_num=3, middle_TC=5, expand=True)
        self.avg = nn.AvgPool2d(kernel_size=2)

        self.spt_conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.temp_cov2 = SpatioTemporalConv2d(self.spt_conv2, time_constants_num=3, middle_TC=5, expand=False)
        self.avg = nn.AvgPool2d(kernel_size=2)

        self.spt_conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.temp_cov3 = SpatioTemporalConv2d(self.spt_conv3, time_constants_num=3, middle_TC=5, expand=False)

        self.spt_conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.temp_cov4 = SpatioTemporalConv2d(self.spt_conv4, time_constants_num=3, middle_TC=5, expand=False)

        self.tc_head = TCHead(16, 3)
        self.warmup = warmup

        self.bn1 = nn.BatchNorm2d(2)
        self.bn2 = nn.BatchNorm2d(8 * 3)
        self.bn3 = nn.BatchNorm2d(8 * 3)
        self.bn4 = nn.BatchNorm2d(8 * 3)
        self.bn5 = nn.BatchNorm2d(16 * 3)

        # add regularization with gaussian blob and frame difference



    def forward(self, x, epoch):
        x_start = x[:, :self.warmup]
        x_end = x[:, self.warmup:]
        self.temp_cov1.reset_state()
        self.temp_cov2.reset_state()
        self.temp_cov3.reset_state()
        self.temp_cov4.reset_state()
        with torch.no_grad():
            _ = self._run_forward(x_start, epoch)

        heatmaps, coords = self._run_forward(x_end, epoch)

        return heatmaps, coords

    def _run_forward(self, x, epoch=None):
        x = x.permute(1, 0, 2, 3, 4)
        """
        x: (T, B, 2, H, W)
        returns:
            heatmaps: (T, B, 3, H, W)
            coords:   (T, B, 3, 2)
        """

        # --- First spatiotemporal block ---
        # apply pooling per timestep
        T, B, C, H, W = x.shape
        x = x.contiguous().view(T * B, C, H, W)
        x = self.bn1(x)
        x = self.avg(x)
        _, _, H, W = x.shape
        x = x.view(T, B, C, H, W)
        # --- Second spatiotemporal block ---
       
        x = self.temp_cov1(x)
        T, B, C, H, W = x.shape
        x = x.view(T * B, C, H, W)
        x = self.bn2(x)
        x = x.view(T, B, C, H, W)

        x = self.temp_cov2(x)
        T, B, C, H, W = x.shape
        x = x.view(T * B, C, H, W)
        x = self.bn3(x)
        x = x.view(T, B, C, H, W)

        x = self.temp_cov3(x)
        T, B, C, H, W = x.shape
        x = x.view(T * B, C, H, W)
        x = self.bn4(x)
        x = x.view(T, B, C, H, W)


        # apply pooling again
        T, B, C, H, W = x.shape
        x = x.view(T * B, C, H, W)
        x = self.avg(x)
        _, _, H, W = x.shape
        x = x.view(T, B, C, H, W)

        x = self.temp_cov4(x)

        T, B, C, H, W = x.shape
        x = x.view(T * B, C, H, W)
        x = self.bn5(x)
        x = x.view(T, B, C, H, W)
        # --- Head ---
        heatmaps, coords = self.tc_head(x, epoch=epoch)
        return heatmaps, coords
    


def make_gaussian_targets(y, W, H, sigma=2.0):
    """
    y: (T, B, C, 2)  -> (x, y) coords
    returns: (T, B, C, W, H)
    """
    T, B, C, _ = y.shape
    device = y.device

    # Create coordinate grid
    xs = torch.arange(W, device=device).view(1, 1, 1, W, 1)
    ys = torch.arange(H, device=device).view(1, 1, 1, 1, H)

    # Ground truth coords
    x0 = y[..., 0].unsqueeze(-1).unsqueeze(-1)  # (T,B,C,1,1)
    y0 = y[..., 1].unsqueeze(-1).unsqueeze(-1)

    # Gaussian
    g = torch.exp(-((xs - x0)**2 + (ys - y0)**2) / (2 * sigma**2))

    return g