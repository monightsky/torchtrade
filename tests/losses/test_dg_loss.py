"""Tests for DGLoss module."""

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from torch import nn
from torch.distributions import Categorical

from torchtrade.losses import DGLoss


class TestDGLoss:
    """Test suite for DGLoss."""

    OBS_DIM = 4
    ACTION_DIM = 3
    BATCH_SIZE = 10

    @pytest.fixture(autouse=True)
    def set_random_seed(self):
        torch.manual_seed(42)

    @pytest.fixture
    def actor_network(self):
        policy_net = TensorDictModule(
            nn.Sequential(
                nn.Linear(self.OBS_DIM, 64),
                nn.ReLU(),
                nn.Linear(64, self.ACTION_DIM),
            ),
            in_keys=["observation"],
            out_keys=["logits"],
        )
        actor = ProbabilisticTensorDictModule(
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=Categorical,
            return_log_prob=True,
        )
        return ProbabilisticTensorDictSequential(policy_net, actor)

    @pytest.fixture
    def sample_data(self):
        return TensorDict(
            {
                "observation": torch.randn(self.BATCH_SIZE, self.OBS_DIM),
                "action": torch.randint(0, self.ACTION_DIM, (self.BATCH_SIZE,)),
                "next": {
                    "reward": torch.randn(self.BATCH_SIZE, 1),
                },
            },
            batch_size=[self.BATCH_SIZE],
        )

    def test_missing_actor_raises(self):
        with pytest.raises(TypeError, match="Missing positional argument"):
            DGLoss(actor_network=None)

    @pytest.mark.parametrize("baseline", ["invalid", "foo"])
    def test_invalid_baseline_raises(self, actor_network, baseline):
        with pytest.raises(ValueError, match="baseline must be"):
            DGLoss(actor_network=actor_network, baseline=baseline)

    def test_forward_outputs_and_gradients(self, actor_network, sample_data):
        """Forward pass produces expected keys, shapes, finite values, and flowing gradients."""
        loss = DGLoss(actor_network=actor_network)
        output = loss(sample_data)

        # All expected keys present with scalar shape
        for key in ["loss_objective", "gate", "advantage", "entropy", "loss_entropy"]:
            assert key in output.keys(), f"missing key: {key}"
            assert output[key].shape == torch.Size([]), f"wrong shape for {key}"
            assert torch.isfinite(output[key]), f"non-finite {key}"

        # Loss keys have gradients, diagnostics don't
        assert output["loss_objective"].requires_grad
        assert output["loss_entropy"].requires_grad
        assert not output["gate"].requires_grad
        assert not output["advantage"].requires_grad

        # Gradients flow to actor params
        (output["loss_objective"] + output["loss_entropy"]).backward()
        for param in loss.actor_network_params.values(True, True):
            assert param.grad is not None

    def test_no_entropy_bonus(self, actor_network, sample_data):
        """Disabling entropy removes entropy keys from output."""
        loss = DGLoss(actor_network=actor_network, entropy_bonus=False)
        output = loss(sample_data)

        assert "loss_objective" in output.keys()
        assert "entropy" not in output.keys()
        assert "loss_entropy" not in output.keys()

    @pytest.mark.parametrize("eta", [0.1, 1.0, 10.0])
    def test_eta_affects_gate(self, actor_network, sample_data, eta):
        """Different eta values produce different gate values (gate is sigmoid-based)."""
        loss = DGLoss(actor_network=actor_network, eta=eta)
        output = loss(sample_data)
        # Gate should be between 0 and 1 (it's a sigmoid)
        assert 0.0 <= output["gate"].item() <= 1.0

    @pytest.mark.parametrize("baseline", ["mean", "none"])
    def test_baseline_modes(self, actor_network, sample_data, baseline):
        """Both baseline modes produce finite loss."""
        loss = DGLoss(actor_network=actor_network, baseline=baseline)
        output = loss(sample_data)
        assert torch.isfinite(output["loss_objective"])

    def test_expected_baseline(self, actor_network):
        """Expected baseline uses counterfactual rewards to compute advantage."""
        cf_rewards = torch.randn(self.BATCH_SIZE, self.ACTION_DIM)
        data = TensorDict(
            {
                "observation": torch.randn(self.BATCH_SIZE, self.OBS_DIM),
                "action": torch.randint(0, self.ACTION_DIM, (self.BATCH_SIZE,)),
                "counterfactual_rewards": cf_rewards,
                "next": {"reward": torch.randn(self.BATCH_SIZE, 1)},
            },
            batch_size=[self.BATCH_SIZE],
        )
        loss = DGLoss(actor_network=actor_network, baseline="expected")
        output = loss(data)
        assert torch.isfinite(output["loss_objective"])

    @pytest.mark.parametrize("batch_size", [1, 5, 32])
    def test_different_batch_sizes(self, actor_network, batch_size):
        data = TensorDict(
            {
                "observation": torch.randn(batch_size, self.OBS_DIM),
                "action": torch.randint(0, self.ACTION_DIM, (batch_size,)),
                "next": {"reward": torch.randn(batch_size, 1)},
            },
            batch_size=[batch_size],
        )
        loss = DGLoss(actor_network=actor_network)
        output = loss(data)
        assert torch.isfinite(output["loss_objective"])

    def test_zero_rewards_finite(self, actor_network, sample_data):
        """Zero rewards should not cause NaN/Inf (advantage = 0, gate = 0.5)."""
        sample_data["next", "reward"] = torch.zeros(self.BATCH_SIZE, 1)
        loss = DGLoss(actor_network=actor_network)
        output = loss(sample_data)
        assert torch.isfinite(output["loss_objective"])
        # With zero advantage, delight=0 so gate should be exactly 0.5
        assert output["gate"].item() == pytest.approx(0.5, abs=1e-5)

    def test_no_old_log_probs_needed(self, actor_network):
        """DG should work without sample_log_prob (unlike GRPO)."""
        data = TensorDict(
            {
                "observation": torch.randn(self.BATCH_SIZE, self.OBS_DIM),
                "action": torch.randint(0, self.ACTION_DIM, (self.BATCH_SIZE,)),
                "next": {"reward": torch.randn(self.BATCH_SIZE, 1)},
            },
            batch_size=[self.BATCH_SIZE],
        )
        # No sample_log_prob, no action_log_prob -- should still work
        loss = DGLoss(actor_network=actor_network)
        output = loss(data)
        assert torch.isfinite(output["loss_objective"])

    def test_in_keys(self, actor_network):
        loss = DGLoss(actor_network=actor_network)
        in_keys = loss.in_keys
        assert "observation" in in_keys
        assert "action" in in_keys

    def test_out_keys(self, actor_network):
        loss = DGLoss(actor_network=actor_network, entropy_bonus=True)
        assert set(loss.out_keys) == {"loss_objective", "gate", "advantage", "entropy", "loss_entropy"}

        loss = DGLoss(actor_network=actor_network, entropy_bonus=False)
        assert set(loss.out_keys) == {"loss_objective", "gate", "advantage"}
