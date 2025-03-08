# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

""" Fast Matrix Exponential Methods

This module implements matrix-free methods for approximating the action of a matrix exponential
on a vector via Krylov subspace techniques. It provides an implementation of the Lanczos iteration
to generate an orthonormal basis for the Krylov subspace, and uses this basis to compute an
approximation of exp(-1j * dt * A) * v without explicitly constructing the matrix A.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.linalg import eigh_tridiagonal

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _lanczos_iteration(
    Afunc: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
    vstart: NDArray[np.complex128],
    numiter: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]:
    """Perform a matrix-free Lanczos iteration.

    This function generates an orthonormal basis for the Krylov subspace of the operator defined
    by Afunc using the Lanczos algorithm. It computes the diagonal (alpha) and off-diagonal (beta)
    elements of the tridiagonal (Hessenberg) matrix associated with the iteration and returns the
    matrix whose columns are the Lanczos vectors.

    Args:
        Afunc (Callable[[NDArray[np.complex128]], NDArray[np.complex128]]):
            A function that applies a linear transformation to a vector without explicitly constructing
            the matrix (i.e., matrix-free).
        vstart (NDArray[np.complex128]):
            The starting vector for the Lanczos iteration. This vector is normalized in place.
        numiter (int):
            The number of Lanczos iterations to perform. It should be much smaller than the dimension of
            vstart.

    Returns:
        tuple:
            - alpha (NDArray[np.float64]): Array of length numiter containing the diagonal entries of the
              tridiagonal matrix.
            - beta (NDArray[np.float64]): Array of length numiter-1 containing the off-diagonal entries.
            - V (NDArray[np.complex128]): A matrix of shape (len(vstart) x numiter) whose columns are the
              orthonormal Lanczos vectors.
    """
    # normalize starting vector
    nrmv = np.linalg.norm(vstart)
    assert nrmv > 0
    vstart /= nrmv

    alpha = np.zeros(numiter)
    beta = np.zeros(numiter - 1)

    V = np.zeros((numiter, len(vstart)), dtype=complex)
    V[0] = vstart

    for j in range(numiter - 1):
        w = Afunc(V[j])
        alpha[j] = np.vdot(w, V[j]).real
        w -= alpha[j] * V[j] + (beta[j - 1] * V[j - 1] if j > 0 else 0)
        beta[j] = np.linalg.norm(w)
        if beta[j] < 100 * len(vstart) * np.finfo(float).eps:
            # Terminate early if the next vector is (numerically) zero.
            numiter = j + 1
            return (alpha[:numiter], beta[: numiter - 1], V[:numiter, :].T)
        V[j + 1] = w / beta[j]

    # Complete final iteration
    j = numiter - 1
    w = Afunc(V[j])
    alpha[j] = np.vdot(w, V[j]).real
    return (alpha, beta, V.T)


def expm_krylov(
    Afunc: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
    v: NDArray[np.complex128],
    dt: float,
    numiter: int,
) -> NDArray[np.complex128]:
    """Compute the Krylov subspace approximation of the matrix exponential applied to a vector.

    This function approximates exp(-1j * dt * A) * v by projecting the action of the matrix exponential
    onto a Krylov subspace generated by the Lanczos iteration. The method is based on the approach
    described by Hochbruck and Lubich.

    Args:
        Afunc (Callable[[NDArray[np.complex128]], NDArray[np.complex128]]):
            A function implementing the matrix-free application of the linear operator A.
        v (NDArray[np.complex128]):
            The input vector to which the matrix exponential is applied.
        dt (float):
            The time step (or scalar multiplier) in the exponential.
        numiter (int):
            The number of Lanczos iterations (and the dimension of the Krylov subspace) to use.

    Returns:
        NDArray[np.complex128]:
            The approximate result of applying exp(-1j * dt * A) to v.
    """
    alpha, beta, V = _lanczos_iteration(Afunc, v, numiter)
    try:
        # Attempt a faster eigen-decomposition for the tridiagonal matrix.
        w_hess, u_hess = eigh_tridiagonal(alpha, beta)
    except Exception:
        # Fall back to a more stable eigen-decomposition if needed.
        w_hess, u_hess = eigh_tridiagonal(alpha, beta, lapack_driver="stebz")
    # Construct the approximation: scale the exponential of the eigenvalues by the norm of v,
    # and project back to the full space via the Lanczos basis V.
    return V @ (u_hess @ (np.linalg.norm(v) * np.exp(-1j * dt * w_hess) * u_hess[0]))
