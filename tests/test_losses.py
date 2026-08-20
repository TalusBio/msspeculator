"""Per-row label masking: a source that reports one RT label and not the other."""

from __future__ import annotations

import math

import torch

from pepdistill.distill.losses import labeled_mse


def test_an_unlabeled_row_cannot_poison_the_gradient():
    """The reason the target is substituted before the subtraction rather than masked after it:
    ``0 * NaN`` is NaN, so masking the residual would leave every shared parameter upstream of
    the RT head with a NaN gradient -- one library row would destroy the whole batch."""
    prediction = torch.tensor([1.0, 5.0, 3.0], requires_grad=True)
    target = torch.tensor([0.0, float("nan"), 1.0])

    loss = labeled_mse(prediction, target)
    loss.backward()

    assert float(loss.detach()) == 1.0 * 0.5 + 4.0 * 0.5  # mean over the two labeled rows
    assert torch.isfinite(prediction.grad).all()
    assert float(prediction.grad[1]) == 0.0  # the unlabeled row contributes nothing


def test_a_batch_with_no_labels_yields_a_differentiable_zero():
    """A batch drawn entirely from a library has no iRT at all. The term has to stay addable
    and stay in the graph, so callers never branch on which labels a batch happens to carry."""
    prediction = torch.tensor([1.0, 5.0], requires_grad=True)
    loss = labeled_mse(prediction, torch.tensor([float("nan")] * 2))

    assert float(loss.detach()) == 0.0
    loss.backward()
    assert float(prediction.grad.abs().sum()) == 0.0


def test_a_nan_prediction_still_surfaces():
    """Masking is keyed on the target. A NaN prediction is a fault and must not be swallowed
    by the same mechanism that tolerates a missing label."""
    loss = labeled_mse(torch.tensor([float("nan"), 2.0]), torch.tensor([0.0, 1.0]))
    assert math.isnan(float(loss))
