"""Information-theoretic PID estimators for multimodal representations.

This package implements the two estimators of partial information
decomposition (PID) proposed in:

    Liang, Cheng, Fan, Hauptmann, Salakhutdinov, Morency,
    "Quantifying & Modeling Multimodal Interactions: An Information
    Decomposition Framework", https://arxiv.org/abs/2302.12247

Given two views ``X1`` and ``X2`` of a sample with shared label ``Y``,
the PID decomposes the joint mutual information ``I(X1, X2; Y)`` into
four non-negative components

    I(X1, X2; Y) = R + U1 + U2 + S

where ``R`` is redundancy (information that either view alone carries),
``U1`` / ``U2`` are the unique informations carried only by one view,
and ``S`` is synergy (information that requires both views jointly).

Public API
----------
- :func:`estimate_pid` — high-level dispatch (``method='cvxpy'|'batch'``).
- :func:`cvxpy_pid`   — discrete cluster-then-CVX estimator.
- :func:`batch_pid`   — neural CE-alignment estimator.
- :class:`PIDResult`  — namedtuple holding the four components and
  the auxiliary mutual information terms.
"""

from src.mutualinfo._common import PIDResult
from src.mutualinfo.batch_estimator import batch_pid
from src.mutualinfo.cvxpy_estimator import cvxpy_pid


def estimate_pid(
    z1,
    z2,
    y,
    method: str = 'cvxpy',
    **kwargs,
) -> PIDResult:
    """Estimate PID components of ``I(Z1, Z2; Y)``.

    Parameters
    ----------
    z1, z2 : array-like, shape (N, d1) / (N, d2)
        Continuous representations of the two views.
    y : array-like, shape (N,)
        Discrete labels.
    method : {'cvxpy', 'batch'}
        Which estimator to use.
    **kwargs
        Forwarded to the underlying estimator.

    Returns
    -------
    PIDResult
    """
    method = method.lower()
    if method in ('cvxpy', 'cvx', 'discrete'):
        return cvxpy_pid(z1, z2, y, **kwargs)
    if method in ('batch', 'neural', 'ce'):
        return batch_pid(z1, z2, y, **kwargs)
    raise ValueError(
        f"Unknown PID method '{method}'. Valid: ['cvxpy', 'batch']."
    )


__all__ = ['PIDResult', 'batch_pid', 'cvxpy_pid', 'estimate_pid']
