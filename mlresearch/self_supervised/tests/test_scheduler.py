import torch
from torch import nn
from torch.optim import Adam

from .._byol import LinearWarmupCosineAnnealingLR


def test_lr_decay():
    scheduler1 = LinearWarmupCosineAnnealingLR(
        Adam(nn.Linear(10, 1).parameters(), lr=0.02), warmup_epochs=10, max_epochs=40
    )
    scheduler2 = LinearWarmupCosineAnnealingLR(
        Adam(nn.Linear(10, 1).parameters(), lr=0.02), warmup_epochs=10, max_epochs=40
    )

    # the default case
    default_lr = []
    for epoch in range(40):
        scheduler1.step()
        default_lr.append(scheduler1.get_last_lr())

    # passing epoch param case
    epoch_lr = []
    for epoch in range(1, 41):
        scheduler2.step(epoch)
        epoch_lr.append(scheduler2.get_last_lr())

    is_decreasing = [
        default_lr[i] < default_lr[i-1] for i in range(1, len(default_lr))
    ]

    torch.testing.assert_allclose(default_lr, epoch_lr)
    assert not all(is_decreasing[:9])
    assert all(is_decreasing[9:])
