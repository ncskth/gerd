import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gerd'))

import unittest
import torch
from render import RenderParameters, render_shape, IAFSubtractReset, ZERO_DISTRIBUTION
import shapes

DEVICE = "cpu"


def near_zero_iaf_init():
    """IAF initial state near zero – no warmup transients leak through."""
    return torch.distributions.Uniform(0.0, 1e-6)


class TestRender(unittest.TestCase):

    # ---------------------------------------------------------------------------
    # Test 1 – zero activations with zero movement
    # ---------------------------------------------------------------------------

    def test_zero_activations_no_movement(self):
        """A static, fully-filled square with no noise must produce zero events.

        shape_density=1 → Bernoulli(1) → all pixels always active, so the diff
        between any two consecutive frames is zero.  The IAF state starts near
        zero and receives no input, so it never crosses the ±0.5 threshold.
        Background noise is disabled.  Expected: images.sum() == 0.
        """
        p = RenderParameters(
            resolution=torch.Size([10, 10]),
            length=5,
            bg_noise_density=0,
            event_density=1,
            shape_density=1,
            polarity=True,
            warmup_steps=2,
            upsampling_factor=4,
            upsampling_cutoff=0.5,
            device=DEVICE,
            initial_integration_distribution=near_zero_iaf_init(),
            translate_start=(5.0, 5.0),   # fixed position at centre
            translate_velocity=0.0,        # no movement
            scale_start=None,              # no scaling
            rotate_start=None,             # no rotation
            shear_start=None,              # no shear
        )
        images, _ = render_shape(shapes.square, p)
        self.assertEqual(images.sum().item(), 0,
            f"Expected zero events for a static scene, got {images.sum().item()}")


    # ---------------------------------------------------------------------------
    # Test 2 – events only at the border when scaling by 1 px/frame
    # ---------------------------------------------------------------------------

    def test_border_events_when_scaling(self):
        """Growing a solid square by 1 px/frame produces events only at the border.

        shape_density=1 → interior pixels are always fully covered, so their
        diff is zero and the IAF never fires there.  Events appear only where
        new pixels are exposed at the growing edge.
        """
        p = RenderParameters(
            resolution=torch.Size([20, 20]),
            length=8,
            bg_noise_density=0,
            event_density=1,
            shape_density=1,
            polarity=True,
            warmup_steps=1,
            upsampling_factor=4,
            upsampling_cutoff=0.5,
            device=DEVICE,
            initial_integration_distribution=near_zero_iaf_init(),
            translate_start=(10.0, 10.0),  # fixed at centre of 20×20 frame
            translate_velocity=0.0,
            scale_start=6.0,               # 6 native px → occupies rows/cols 7–12
            scale_velocity=1.0,            # grow 1 native px per frame
            rotate_start=None,
            shear_start=None,
        )
        images, _ = render_shape(shapes.square, p)

        # There should be events (shape is growing)
        self.assertGreater(images.sum().item(), 0,
            "Expected nonzero events while scaling")

        # The 2×2 centre of the square is always fully covered → no events
        cx, cy = 10, 10
        centre = images[:, :, cx-1:cx+1, cy-1:cy+1]
        self.assertEqual(centre.sum().item(), 0,
            f"Expected no events in the interior, got {centre.sum().item()}")


    # ---------------------------------------------------------------------------
    # Test 3 – expected event count from background noise
    # ---------------------------------------------------------------------------

    def test_background_noise_event_count(self):
        """bg_noise_density=0.01 on a 10×10 frame gives ~1 event/frame on average.

        With polarity=True the implementation halves the rate per channel:
          E[events/frame] = 2 channels × 100 pixels × 0.005 = 1.0
        Shape events are suppressed (static scene, near-zero IAF init).
        Over 500 frames the sample mean should be within ±50 % of 1.0.
        """
        torch.manual_seed(0)
        n_frames = 500
        p = RenderParameters(
            resolution=torch.Size([10, 10]),
            length=n_frames,
            bg_noise_density=0.01,
            event_density=1,
            shape_density=1,
            polarity=True,
            warmup_steps=2,
            upsampling_factor=4,
            upsampling_cutoff=0.5,
            device=DEVICE,
            initial_integration_distribution=near_zero_iaf_init(),
            translate_start=(5.0, 5.0),
            translate_velocity=0.0,
            scale_start=None,
            rotate_start=None,
            shear_start=None,
        )
        images, _ = render_shape(shapes.square, p)
        mean_per_frame = images.sum().item() / n_frames
        self.assertAlmostEqual(mean_per_frame, 1.0, delta=0.5,
            msg=f"Expected ~1 event/frame from bg noise, got {mean_per_frame:.3f}")


    # ---------------------------------------------------------------------------
    # Test 4 – expected event count from event-density (shape sampling) noise
    # ---------------------------------------------------------------------------

    def test_event_density_noise_count(self):
        """event_density=0.01 passes ~1 % of detected diff events.

        We render the same moving scene twice — once with full event density
        and once with 1 % density — and check that the ratio is near 0.01.
        Uses a fixed seed so both runs see the same shape sequence.
        """
        base = dict(
            resolution=torch.Size([40, 40]),
            length=200,
            bg_noise_density=0,
            shape_density=1,
            polarity=True,
            warmup_steps=2,
            upsampling_factor=4,
            upsampling_cutoff=0.5,
            device=DEVICE,
            initial_integration_distribution=near_zero_iaf_init(),
            translate_start=(20.0, 20.0),
            translate_velocity=(1.0, 0.0),   # 1 px/frame → steady stream of events
            scale_start=None,
            rotate_start=None,
            shear_start=None,
        )

        torch.manual_seed(0)
        images_full, _ = render_shape(shapes.square,
                                      RenderParameters(**base, event_density=1.0))
        torch.manual_seed(0)
        images_sparse, _ = render_shape(shapes.square,
                                        RenderParameters(**base, event_density=0.01))

        total_full = images_full.sum().item()
        total_sparse = images_sparse.sum().item()
        self.assertGreater(total_full, 0, "Full-density run produced no events")
        ratio = total_sparse / total_full
        self.assertAlmostEqual(ratio, 0.01, delta=0.005,
            msg=f"Expected ~1 % event ratio, got {ratio:.4f}")


    # ---------------------------------------------------------------------------
    # Test 5 – upsampling works as intended (10×10 × 8 → internal 80×80)
    # ---------------------------------------------------------------------------

    def test_upsampling(self):
        """Output always stays at native resolution regardless of upsampling_factor.

        A 40×40 frame upsampled by 8 uses an internal 320×320 canvas; by 4 a
        160×160 canvas; by 1 a 40×40 canvas.  All three must return tensors
        of shape (length, 2, 40, 40).  We also verify that motion does produce
        events when using factor=8, confirming the internal pipeline runs
        end-to-end.
        """
        resolution = torch.Size([40, 40])
        length = 20

        for uf in [1, 4, 8]:
            p = RenderParameters(
                resolution=resolution,
                length=length,
                bg_noise_density=0,
                event_density=1,
                shape_density=1,
                polarity=True,
                warmup_steps=2,
                upsampling_factor=uf,
                upsampling_cutoff=0.5,
                device=DEVICE,
                initial_integration_distribution=near_zero_iaf_init(),
                translate_start=(20.0, 20.0),
                translate_velocity=(1.0, 0.0),
                scale_start=None,
                rotate_start=None,
                shear_start=None,
            )
            images, labels = render_shape(shapes.square, p)
            self.assertEqual(images.shape, (length, 2, *resolution),
                f"Wrong output shape for upsampling_factor={uf}")
            self.assertEqual(labels.shape, (length, 2),
                f"Wrong label shape for upsampling_factor={uf}")

        # Confirm that 1 px/frame motion with factor=8 actually produces events
        p8 = RenderParameters(
            resolution=resolution, length=length,
            bg_noise_density=0, event_density=1, shape_density=1, polarity=True,
            warmup_steps=2, upsampling_factor=8, upsampling_cutoff=0.5,
            device=DEVICE, initial_integration_distribution=near_zero_iaf_init(),
            translate_start=(20.0, 20.0), translate_velocity=(1.0, 0.0),
            scale_start=None, rotate_start=None, shear_start=None,
        )
        images8, _ = render_shape(shapes.square, p8)
        self.assertGreater(images8.sum().item(), 0,
            "Expected nonzero events for 1 px/frame motion with upsampling_factor=8")


    # ---------------------------------------------------------------------------
    # Test 6 – IAF neuron (IAFSubtractReset)
    # ---------------------------------------------------------------------------

    def test_iaf_neuron(self):
        """IAFSubtractReset integrates, fires, and resets correctly.

        Threshold = 0.5, state starts at 0.
          * input  0.6 > +0.5 → positive spike,  state resets to 0.6 - 0.5 = 0.1
          * input -0.7 < -0.5 → negative spike,  state resets to -0.7 + 0.5 = -0.2
          * input  0.3 → no spike (0.3 < 0.5),   state accumulates to 0.3
          * input  0.3 again → fires (0.3+0.3=0.6 > 0.5), state = 0.6 - 0.5 = 0.1
        """
        cutoff = 0.5
        iaf = IAFSubtractReset(
            cutoff, torch.distributions.Uniform(0.0, 1e-9)
        )
        zero_state = torch.zeros(1, 1)

        # Positive spike
        events, state = iaf(torch.tensor([[0.6]]), zero_state)
        self.assertTrue(events[0].item())           # pos channel fired
        self.assertFalse(events[1].item())          # neg channel silent
        self.assertAlmostEqual(state.item(), 0.1, places=5)

        # Negative spike
        events, state = iaf(torch.tensor([[-0.7]]), zero_state)
        self.assertFalse(events[0].item())
        self.assertTrue(events[1].item())           # neg channel fired
        self.assertAlmostEqual(state.item(), -0.2, places=5)

        # No spike
        events, state = iaf(torch.tensor([[0.3]]), zero_state)
        self.assertFalse(events[0].item())
        self.assertFalse(events[1].item())
        self.assertAlmostEqual(state.item(), 0.3, places=5)

        # Accumulate across two steps: 0.3 + 0.3 = 0.6 > 0.5 → fires
        events, state = iaf(torch.tensor([[0.3]]), state)
        self.assertTrue(events[0].item())
        self.assertAlmostEqual(state.item(), 0.1, places=5)


    # ---------------------------------------------------------------------------
    # Test 7 – all three shapes produce correctly shaped data with nonzero events
    # ---------------------------------------------------------------------------

    def test_all_shapes_nonzero_output(self):
        """circle, triangle, and square each return the right tensor shape and
        fire at least some events when the shape translates across the frame.
        """
        resolution = torch.Size([40, 40])
        length = 20

        for shape_fn in [shapes.circle, shapes.triangle, shapes.square]:
            with self.subTest(shape=shape_fn.__name__):
                p = RenderParameters(
                    resolution=resolution,
                    length=length,
                    bg_noise_density=0.001,
                    event_density=1,
                    shape_density=1,
                    polarity=True,
                    warmup_steps=2,
                    upsampling_factor=4,
                    upsampling_cutoff=0.5,
                    device=DEVICE,
                    translate_start=(20.0, 20.0),
                    translate_velocity=(1.0, 0.5),
                    scale_start=None,
                    rotate_start=None,
                    shear_start=None,
                )
                images, labels = render_shape(shape_fn, p)
                self.assertEqual(images.shape, (length, 2, *resolution),
                    f"Wrong events shape for {shape_fn.__name__}")
                self.assertEqual(labels.shape, (length, 2),
                    f"Wrong labels shape for {shape_fn.__name__}")
                self.assertGreater(images.sum().item(), 0,
                    f"Expected nonzero events for {shape_fn.__name__}")


if __name__ == "__main__":
    unittest.main()
