"""Keep the deterministic contract suite on a single CPU thread."""

from collections.abc import Iterator

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def cpu_threads() -> Iterator[None]:
    """Avoid oversubscription for tiny test tensors and restore the setting."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
