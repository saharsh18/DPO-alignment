import unittest
import torch
import math
from dpo.loss import bradley_terry, dpo_loss, ipo_loss
from dpo.metrics import sequence_logprob, reward_metrics, token_kl
from dpo.dataset import VOCAB, make_preference_pairs, encode_chat, encode_pairs

class TestDPO(unittest.TestCase):
    def test_bradley_terry(self):
        p = bradley_terry(torch.tensor(2.0), torch.tensor(0.0))
        self.assertAlmostEqual(p.item(), 0.8808, places=4)
        self.assertEqual(bradley_terry(torch.tensor(1.0), torch.tensor(1.0)).item(), 0.5)

    def test_dpo_loss(self):
        same = torch.tensor([-3.0, -5.0])
        loss, cr, rr = dpo_loss(same, same, same, same, 1.0)
        self.assertAlmostEqual(loss.item(), math.log(2), places=5)

    def test_ipo_loss(self):
        loss_zero = ipo_loss(torch.tensor([5.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), 0.1)
        self.assertEqual(loss_zero.item(), 0.0)

    def test_sequence_logprob(self):
        toy_logits = torch.zeros(1, 3, 4)
        toy_logits[0, 0, 2] = math.log(3.0)
        toy = sequence_logprob(toy_logits, torch.tensor([[0, 2, 1]]), torch.tensor([[0, 1, 1]]))
        self.assertAlmostEqual(toy.item(), math.log(0.5) + math.log(0.25), places=5)

    def test_reward_metrics(self):
        m = reward_metrics(torch.tensor([1.0, 0.0, 2.0, 0.5]), torch.tensor([0.0, 0.0, 3.0, -0.5]))
        self.assertAlmostEqual(m["accuracy"], 0.5, places=5)
        self.assertAlmostEqual(m["margin"], 0.25, places=5)

    def test_vocab_and_pairs(self):
        self.assertEqual(len(VOCAB), 95)
        self.assertEqual(len(make_preference_pairs()), 600)

if __name__ == '__main__':
    unittest.main()
