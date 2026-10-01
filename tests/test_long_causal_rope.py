import unittest
from unittest.mock import patch

import torch

# Import the native package on CPU without querying the unrelated T5
# encoder's default CUDA device. No attention kernel or RoPE code is patched.
with patch('torch.cuda.current_device', return_value=0):
    from wan.modules.model import rope_params
    from wan.modules.model_causal import rope_apply
    from wan.modules.model_fast import causal_rope_apply


def frequency_table(length, head_dim, theta):
    c = head_dim // 2
    dimensions = [c - 2 * (c // 3), c // 3, c // 3]
    return torch.cat([rope_params(length, 2 * dim, theta=theta) for dim in dimensions], dim=1)


def rotation_oracle(x, grid, start, theta):
    frames, height, width = grid
    c = x.shape[-1] // 2
    dimensions = [c - 2 * (c // 3), c // 3, c // 3]
    tt, yy, xx = torch.meshgrid(
        torch.arange(start, start + frames, dtype=torch.float64),
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64), indexing='ij',
    )
    phases = []
    for coord, dimension in zip((tt, yy, xx), dimensions):
        frequency = theta ** (-torch.arange(dimension, dtype=torch.float64) / dimension)
        phases.append(coord.reshape(-1, 1) * frequency)
    phase = torch.cat(phases, -1).unsqueeze(1)
    pairs = x[0, :frames * height * width].reshape(-1, x.shape[2], c, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((a * phase.cos() - b * phase.sin(), a * phase.sin() + b * phase.cos()), -1).flatten(-2)
    return torch.cat((rotated, x[0, frames * height * width:]), dim=0).unsqueeze(0)


class LongCausalRopeTests(unittest.TestCase):
    def test_absolute_phases_across_and_beyond_the_table_boundary(self):
        torch.manual_seed(3)
        grid = (3, 2, 2)
        for function in (rope_apply, causal_rope_apply):
            for head_dim in (12, 20):
                for theta in (256., 10000.):
                    x = torch.randn(1, 14, 2, head_dim, dtype=torch.float64)
                    table = frequency_table(1024, head_dim, theta)
                    for start in (0, 1022, 1024, 16384):
                        with self.subTest(function=function.__name__, dim=head_dim, theta=theta, start=start):
                            actual = function(x, torch.tensor([grid]), table, start_frame=start)
                            expected = rotation_oracle(x, grid, start, theta)
                            torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)

    def test_rotation_preserves_norms_and_padding_gradients(self):
        grid = (3, 1, 1)
        for function in (rope_apply, causal_rope_apply):
            with self.subTest(function=function.__name__):
                x = torch.randn(1, 5, 2, 12, dtype=torch.float64, requires_grad=True)
                actual = function(x, torch.tensor([grid]), frequency_table(1024, 12, 10000.), start_frame=4096)
                torch.testing.assert_close(actual.square().sum(-1), x.square().sum(-1))
                torch.testing.assert_close(actual[:, 3:], x[:, 3:])
                actual.square().sum().backward()
                torch.testing.assert_close(x.grad, 2. * x.detach())

    def test_large_absolute_offset_is_not_wrapped_to_cached_positions(self):
        x = torch.ones(1, 1, 1, 12, dtype=torch.float64)
        grid = torch.tensor([[1, 1, 1]])
        table = frequency_table(1024, 12, 10000.)
        long_position = rope_apply(x, grid, table, start_frame=4096)
        origin = rope_apply(x, grid, table, start_frame=0)
        self.assertFalse(torch.allclose(long_position, origin))
        torch.testing.assert_close(long_position, rotation_oracle(x, (1, 1, 1), 4096, 10000.))


if __name__ == '__main__':
    unittest.main()
