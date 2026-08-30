import unittest

import torch

from videox_fun.data.bucket_sampler import RandomSampler
from videox_fun.training.sample_loss_recorder import padded_epoch_sample_count


class CompleteBatchSamplerTest(unittest.TestCase):
    def test_prefix_request_rounds_up_to_the_next_effective_batch(self):
        requested = 255
        aligned = ((requested + 32 - 1) // 32) * 32
        self.assertEqual(aligned, 256)
        self.assertEqual(aligned % 32, 0)

    def test_padded_sampler_yields_full_batches_and_complete_coverage(self):
        num_samples = 13
        scheduled_samples = padded_epoch_sample_count(
            num_samples,
            num_processes=2,
            gradient_accumulation_steps=1,
            batch_size=4,
        )
        sampler = RandomSampler(
            range(num_samples),
            replacement=False,
            num_samples=scheduled_samples,
            generator=torch.Generator().manual_seed(42),
        )
        batches = list(torch.utils.data.BatchSampler(sampler, 4, drop_last=False))
        sampled = [index for batch in batches for index in batch]

        self.assertEqual(scheduled_samples, 16)
        self.assertEqual(len(sampled), scheduled_samples)
        self.assertTrue(all(len(batch) == 4 for batch in batches))
        self.assertEqual(set(sampled), set(range(num_samples)))


if __name__ == "__main__":
    unittest.main()
