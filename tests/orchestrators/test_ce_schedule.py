"""Communication schedule of CESheafFRL / CEDiSheafFRL.

Paper, App. ablation (number of communication epochs): K = max(1, round(rho * N)) epochs,
spread uniformly, the last one included.
"""

from types import SimpleNamespace

import pytest

from src.orchestrators import CEDiSheafFRL, CESheafFRL


def schedule(percentage: float, n_epochs: int) -> list[int]:
    """Communicating epochs of a ``percentage`` run over ``n_epochs``, through the class methods."""
    ns = SimpleNamespace(_comm_percentage=percentage, _n_epochs=n_epochs)
    ns._comm_epochs = CESheafFRL._comm_epochs_for(ns, n_epochs)
    return [e for e in range(n_epochs) if CESheafFRL._is_collab_epoch(ns, e)]


@pytest.mark.parametrize('percentage', [1, 2, 5, 10, 20, 40, 60, 80, 100])
def test_count_and_last_epoch(percentage):
    """Over 100 epochs, rho% of them communicate, the last one included."""
    epochs = schedule(percentage, 100)
    assert len(epochs) == percentage  # 100 epochs: rho% of them
    assert epochs[-1] == 99  # the maps always refit on the final encoders


@pytest.mark.parametrize('percentage', [2, 5, 10, 20, 40, 60, 80])
def test_paper_rule_and_uniform_spacing(percentage):
    """The schedule is the paper rule and evenly spaced."""
    n, k = 100, round(percentage / 100 * 100)
    epochs = schedule(percentage, n)
    assert epochs == [e for e in range(n) if (e + 1) * k // n > e * k // n]
    gaps = {b - a for a, b in zip(epochs, epochs[1:])}
    assert max(gaps) - min(gaps) <= 1  # as even as the integers allow


def test_at_least_one_epoch():
    """A tiny share still communicates once, at the last epoch."""
    assert schedule(0.1, 100) == [99]


def test_full_communication_every_epoch():
    """At 100% every epoch communicates."""
    assert schedule(100, 7) == list(range(7))


@pytest.mark.parametrize('cls', [CESheafFRL, CEDiSheafFRL])
@pytest.mark.parametrize('percentage', [0, -5, 101])
def test_out_of_range_percentage_rejected(cls, percentage):
    """Shares outside (0, 100] are refused."""
    with pytest.raises(ValueError, match='comm_percentage'):
        cls(comm_percentage=percentage)


def test_local_reg_rejected_by_ce_di_sheaf_frl():
    """CEDiSheafFRL refuses local_reg (no pilot pass on local epochs)."""
    with pytest.raises(ValueError, match='local_reg'):
        CEDiSheafFRL(comm_percentage=50, local_reg=True)
