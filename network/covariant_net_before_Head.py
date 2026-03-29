from norse.torch import ParameterizedSpatialReceptiveField2d
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


class LI(nn.Module):
    """Stateful Leaky Integrator activation using the existing LeakyIntegrator.
    
    Maintains internal state across timesteps and resets after processing.
    """
    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.li = LeakyIntegrator()
        self.tau = torch.tensor(tau, dtype=torch.float32)
        self.state = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LI activation with state accumulation.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor with accumulated state
        """
        self.state = self.li(x, prev_state=self.state, tau=self.tau)
        return self.state
    
    def reset_state(self):
        """Reset the internal state (call after each datapoint)."""
        self.state = None


class SpatioTemporalConv2d(nn.Module):
    def __init__(
        self,
        conv: nn.Conv2d,
        time_constants_num: int,
        middle_TC: float = 10.0,
        TC_ratio: float = torch.sqrt(torch.tensor(2.0)),
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
    def __init__(self, in_channels, num_TC, kernel_size=11):
        super().__init__()
        self.num_TC = num_TC
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        # Shared temporal convolution
        self.head = nn.Conv2d(in_channels, 16, kernel_size=15, padding=15//2, bias=False)
        self.temp_cov = SpatioTemporalConv2d(self.head, time_constants_num=5, middle_TC=10, expand=False)
        
        # THREE SEPARATE HEADS (one per shape) with two-layer architecture (64 → 32 → 1)
        # Circle head: optimized for smooth, symmetric features
        # self.head_circle = nn.Sequential(
        #     nn.Conv2d(64, 16, kernel_size=15, padding=15//2, bias=False),
        #     LI(tau=1.0),
        #     nn.Conv2d(16, 1, kernel_size=15, padding=15//2, bias=False)
        # )
        
        # # Triangle head: optimized for corners and edges
        # self.head_triangle = nn.Sequential(
        #     nn.Conv2d(64, 16, kernel_size=15, padding=15//2, bias=False),
        #     LI(tau=1.0),
        #     nn.Conv2d(16, 1, kernel_size=15, padding=15//2, bias=False)
        # )
        
        # Square head: optimized for right angles and symmetry
        self.head_square = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=15, padding=15//2, bias=False),
            # LI(tau=1.0),
            nn.Conv2d(16, 1, kernel_size=15, padding=15//2, bias=False)
        )
        
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, epoch=None, curriculum_weights=None):
        """Forward pass through temporal attention head.
        
        Args:
            x: Input of shape (T, B, C*num_TC, H, W)
            epoch: Current epoch (unused, kept for compatibility)
            curriculum_weights: Optional curriculum weights for time constants
            
        Returns:
            heatmaps: Spatial attention maps of shape (T, B, 3, H, W)
            coords: Predicted coordinates of shape (T, B, 3, 2)
        """
        # Apply temporal convolution
        x = self.temp_cov(x)

        # Aggregate across time constants
        T, B, C_total, H, W = x.shape
        C = C_total // self.num_TC
        x = x.view(T, B, self.num_TC, C, H, W)
        
        if curriculum_weights is not None:
            # Apply curriculum weighting instead of mean averaging
            curriculum_weights = curriculum_weights.to(x.device)
            x = (x * curriculum_weights.view(1, 1, -1, 1, 1, 1)).sum(dim=2)
        else:
            # Default: max across time constants (scale-max pooling)
            x = x.mean(dim=2)  # (T, B, C, H, W)
            # x = x.max(dim=2).values  # (T, B, C, H, W)

        # Reset LI states at the beginning of each datapoint
        # for head in [self.head_square]:
        # for head in [self.head_circle, self.head_triangle, self.head_square]:
            # head[1].reset_state()  # Reset LI activation (index 1 in Sequential)
        
        # Process timestep by timestep to accumulate LI state
        heatmaps_list = []
        for t in range(T):
            x_t = x[t]  # (B, C, H, W)
            
            # Apply three shape-specific heads
            # h_circle = self.head_circle(x_t)      # (B, 1, H, W)
            # h_triangle = self.head_triangle(x_t)  # (B, 1, H, W)
            h_square = self.head_square(x_t)      # (B, 1, H, W)
            
            # Concatenate along channel dimension
            # h_t = torch.cat([h_circle, h_triangle, h_square], dim=1)  # (B, 3, H, W)
            h_t = h_square  # (B, 1, H, W)
            heatmaps_list.append(h_t)
        
        # Stack timesteps back together
        heatmaps = torch.stack(heatmaps_list, dim=0)  # (T, B, 1, H, W)
        
        # Reshape for pooling
        heatmaps = heatmaps.view(T * B, 1, H, W)
        heatmaps = self.pool(heatmaps)
        _, _, H, W = heatmaps.shape
        heatmaps = heatmaps.view(T, B, 1, H, W)

        # Extract coordinates from heatmaps via soft-argmax
        coords = self.soft_argmax_2d(heatmaps)

        return heatmaps, coords

    def soft_argmax_2d(self, heatmaps):
        """Compute soft-argmax of heatmaps to extract coordinates.
        
        Args:
            heatmaps: Spatial attention maps of shape (T, B, 3, H, W)
            
        Returns:
            Normalized coordinates (x, y) in [0, 1] of shape (T, B, 3, 2)
        """
        T, B, C, H, W = heatmaps.shape
        heatmaps_flat = heatmaps.view(T, B, C, H * W)

        # Compute probability distribution over spatial locations
        probs = torch.softmax(heatmaps_flat, dim=-1)  # (T, B, C, H*W)

        # Create normalized coordinate grids
        xs = torch.linspace(0, 1, W, device=heatmaps.device)
        ys = torch.linspace(0, 1, H, device=heatmaps.device)
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
    def __init__(self, warmup=10, curriculum=False, total_epochs=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup = warmup
        self.curriculum = curriculum
        self.total_epochs = total_epochs
        
        kernel_size = 11
        time_constants_num = 5
        scales = torch.tensor([1.0, 2.0, 4.0, 8.0])
        angles = torch.tensor([0.0, torch.pi/2])
        ratios = torch.tensor([0.5, 1])
        x = torch.tensor([0.0])
        y = torch.tensor([0.0])
        derivatives = 0
        self.m = ParameterizedSpatialReceptiveField2d(2, kernel_size, scales=scales, angles=angles, ratios=ratios, derivatives=derivatives, x=x, y=y)
        weights = self.m.submodule.weights
        self.spt_conv1 = nn.Conv2d(in_channels=2, out_channels=weights.shape[0], kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)
        with torch.no_grad():
            self.spt_conv1.weight.copy_(weights)
        self.temp_cov1 = SpatioTemporalConv2d(self.spt_conv1, time_constants_num=time_constants_num, middle_TC=10, expand=True)
        self.pool = nn.MaxPool2d(kernel_size=2)

        x = torch.tensor([0.0])
        y = torch.tensor([0.0])
        derivatives = 1

        self.m2 = ParameterizedSpatialReceptiveField2d(weights.shape[0], kernel_size, scales=scales, angles=angles, ratios=ratios, derivatives=derivatives, x=x, y=y)
        weights2 = self.m2.submodule.weights

        self.spt_conv2 = nn.Conv2d(in_channels=weights.shape[0], out_channels=weights2.shape[0], kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)
        with torch.no_grad():
            self.spt_conv2.weight.copy_(weights2)

        self.temp_cov2 = SpatioTemporalConv2d(self.spt_conv2, time_constants_num=time_constants_num, middle_TC=10, expand=False)

        self.tc_head = TCHead(weights2.shape[0], time_constants_num)
        self.num_tc = time_constants_num

    def curriculum_tc_weights(self, epoch, num_tc):
        """
        Compute curriculum learning weights for time constant channels.
        
        Early epochs: Learn only middle channel (one-hot)
        Late epochs: Equal contribution from all channels
        
        Args:
            epoch: Current training epoch
            num_tc: Number of time constant channels
            
        Returns:
            Weighted averaging factors of shape (num_tc,)
        """
        if self.total_epochs <= 0:
            return torch.ones(num_tc) / num_tc
        
        progress = min(epoch / self.total_epochs, 1.0)
        middle_idx = num_tc // 2
        
        # Start with one-hot on middle, transition to uniform
        weights = torch.ones(num_tc) * (progress / num_tc)
        weights[middle_idx] += (1 - progress)
        
        return weights / weights.sum()  # Ensure normalization

    def forward(self, x, epoch=None):
        x_start = x[:, :self.warmup]
        x_end = x[:, self.warmup:]
        self.temp_cov1.reset_state()
        self.temp_cov2.reset_state()
        self.tc_head.temp_cov.reset_state()
        with torch.no_grad():
            _ = self._run_forward(x_start, epoch)

        heatmaps, coords = self._run_forward(x_end, epoch)

        return heatmaps, coords

    def _run_forward(self, x, epoch=None):
        """Forward pass through spatiotemporal network.
        
        Args:
            x: Input tensor of shape (B, T, 2, H, W)
            epoch: Current training epoch (unused, kept for compatibility)
            
        Returns:
            heatmaps: Spatial attention maps of shape (T, B, 3, H, W)
            coords: Predicted coordinates of shape (T, B, 3, 2)
        """
        x = x.permute(1, 0, 2, 3, 4)  # (T, B, 2, H, W)

        # Initial pooling: 300 → 150
        T, B, C, H, W = x.shape
        x = x.contiguous().view(T * B, C, H, W)
        x = self.pool(x)
        _, _, H, W = x.shape
        x = x.view(T, B, C, H, W)

        # First spatiotemporal block: 150 → 150
        x = self.temp_cov1(x)
        T, B, C, H, W = x.shape
        x = x.contiguous().view(T * B, C, H, W)
        x = self.pool(x)  # 150 → 75
        _, _, H, W = x.shape
        x = x.view(T, B, C, H, W)

        # Second spatiotemporal block: 75 → 75 (preserve resolution)
        x = self.temp_cov2(x)
        T, B, C, H, W = x.shape
        x = x.view(T * B, C, H, W)
        x = self.pool(x)
        _, _, H, W = x.shape
        x = x.view(T, B, C, H, W)

        # Head: produces heatmaps and coordinates
        curriculum_weights = None
        if self.curriculum and epoch is not None:
            curriculum_weights = self.curriculum_tc_weights(epoch, self.num_tc)
        heatmaps, coords = self.tc_head(x, epoch=epoch, curriculum_weights=curriculum_weights)
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

    return g.permute(0, 1, 2, 4, 3)