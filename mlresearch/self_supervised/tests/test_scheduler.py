from torch import nn
from torch.optim import Adam

from .._components import LinearWarmupCosineAnnealingLR


def test_lr_decay():
    start_lr = 0.001
    last_lr = 0.0001
    epochs = 40
    scheduler = LinearWarmupCosineAnnealingLR(
        Adam(nn.Linear(10, 1).parameters(), lr=0.02),
        warmup_epochs=10,
        max_epochs=epochs,
        warmup_start_lr=start_lr,
        eta_min=last_lr,
    )
    # the default case
    default_lr = []
    for epoch in range(epochs):
        scheduler.optimizer.step()
        scheduler.step()
        default_lr.append(scheduler.get_last_lr())

    is_decreasing = [
        default_lr[i] < default_lr[i - 1] for i in range(1, len(default_lr))
    ]

    assert not all(is_decreasing[:9])
    assert all(is_decreasing[9:])
    assert start_lr == default_lr[0][0]
    assert last_lr == default_lr[-1][0]
