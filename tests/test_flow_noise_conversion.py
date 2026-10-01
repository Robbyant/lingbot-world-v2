import unittest
from unittest.mock import patch

import torch

# Package initialization evaluates the unrelated T5 encoder's CUDA default.
# Only defer that device query while importing; the schedulers run on CPU.
with patch('torch.cuda.current_device', return_value=0):
    from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def noise_schedulers():
    return [
        FlowDPMSolverMultistepScheduler(algorithm_type='dpmsolver', final_sigmas_type='sigma_min'),
        FlowDPMSolverMultistepScheduler(algorithm_type='sde-dpmsolver', final_sigmas_type='sigma_min'),
        FlowUniPCMultistepScheduler(predict_x0=False),
    ]


class FlowNoiseConversionTests(unittest.TestCase):
    def test_noise_prediction_inverts_the_rectified_flow_path(self):
        data = torch.tensor([[[[0.2, -0.5], [1.3, 0.7]]], [[[0.4, 1.1], [-0.3, 0.6]]]], dtype=torch.float64)
        noise = torch.tensor([[[[1.1, 0.3], [-0.6, -0.2]]], [[[-0.9, 0.4], [0.8, -0.1]]]], dtype=torch.float64)
        velocity = noise - data
        for scheduler in noise_schedulers():
            scheduler.sigmas = torch.tensor([0., 0.05, 0.5, 0.95, 1.], dtype=torch.float64)
            for i, sigma in enumerate(scheduler.sigmas):
                with self.subTest(scheduler=type(scheduler).__name__, sigma=float(sigma)):
                    scheduler._step_index = i
                    sample = (1. - sigma) * data + sigma * noise
                    actual = scheduler.convert_model_output(velocity, sample=sample)
                    torch.testing.assert_close(actual, noise)
                    torch.testing.assert_close(actual - velocity, data)

    def test_first_order_noise_update_matches_the_exact_straight_path(self):
        data = torch.tensor([[[[0.2, -0.5], [1.3, 0.7]]]])
        noise = torch.tensor([[[[1.1, 0.3], [-0.6, -0.2]]]])
        velocity = noise - data
        start, end = 0.8, 0.4
        sample = (1. - start) * data + start * noise
        expected = (1. - end) * data + end * noise
        dpm, _, unipc = noise_schedulers()
        for scheduler in (dpm, unipc):
            with self.subTest(scheduler=type(scheduler).__name__):
                scheduler.sigmas = torch.tensor([start, end])
                scheduler._step_index = 0
                prediction = scheduler.convert_model_output(velocity, sample=sample)
                if isinstance(scheduler, FlowDPMSolverMultistepScheduler):
                    actual = scheduler.dpm_solver_first_order_update(prediction, sample=sample)
                else:
                    scheduler.model_outputs[-1] = prediction
                    scheduler.timestep_list[-1] = scheduler.timesteps[0]
                    actual = scheduler.multistep_uni_p_bh_update(velocity, sample=sample, order=1)
                torch.testing.assert_close(actual, expected)

    def test_default_data_prediction_is_preserved(self):
        data = torch.tensor([[[[0.1, 0.4], [-0.3, 0.8]]]])
        noise = torch.tensor([[[[0.7, -0.2], [0.6, -0.5]]]])
        for scheduler in (FlowDPMSolverMultistepScheduler(), FlowUniPCMultistepScheduler()):
            scheduler.sigmas = torch.tensor([0.6])
            scheduler._step_index = 0
            sample = 0.4 * data + 0.6 * noise
            torch.testing.assert_close(scheduler.convert_model_output(noise - data, sample=sample), data)


if __name__ == '__main__':
    unittest.main()
