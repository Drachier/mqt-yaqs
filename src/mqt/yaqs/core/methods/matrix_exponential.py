# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Fast Matrix Exponential Methods.

This module implements matrix-free methods for approximating the action of a matrix exponential
on a vector via Krylov subspace techniques. It provides an implementation of the Lanczos iteration
to generate an orthonormal basis for the Krylov subspace, and uses this basis to compute an
approximation of exp(-1j * dt * A) * v without explicitly constructing the matrix A.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import eigh_tridiagonal

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def lanczos_iteration(
    matrix_free_operator: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
    vec: NDArray[np.complex128],
    lanczos_iterations: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]:
    """Perform a matrix-free Lanczos iteration.

    This function generates an orthonormal basis for the Krylov subspace of the operator defined
    by matrix_free_operator using the Lanczos algorithm. It computes the diagonal (alpha) and off-diagonal (beta)
    elements of the tridiagonal (Hessenberg) matrix associated with the iteration and returns the
    matrix whose columns are the Lanczos vectors.

    Args:
        matrix_free_operator (Callable[[NDArray[np.complex128]], NDArray[np.complex128]]):
            A function that applies a linear transformation to a vector without explicitly constructing
            the matrix (i.e., matrix-free).
        vec (NDArray[np.complex128]):
            The starting vector for the Lanczos iteration. This vector is normalized in place.
        lanczos_iterations (int):
            The number of Lanczos iterations to perform. It should be much smaller than the dimension of
            vec.

    Returns:
        tuple:
            - alpha (NDArray[np.float64]): Array of length lanczos_iterations containing the diagonal entries of the
              tridiagonal matrix.
            - beta (NDArray[np.float64]): Array of length lanczos_iterations-1 containing the off-diagonal entries.
            - lanczos_mat (NDArray[np.complex128]): A matrix of shape (len(vec) x lanczos_iterations) whose
              columns are the orthonormal Lanczos vectors.
    """
    # normalize starting vector
    vec /= np.linalg.norm(vec)

    alpha = np.zeros(lanczos_iterations)
    beta = np.zeros(lanczos_iterations - 1)

    lanczos_mat = np.zeros((lanczos_iterations, len(vec)), dtype=complex)
    lanczos_mat[0] = vec

    for j in range(lanczos_iterations - 1):
        w_j = matrix_free_operator(lanczos_mat[j])
        alpha[j] = np.vdot(w_j, lanczos_mat[j]).real
        w_j -= alpha[j] * lanczos_mat[j] + (beta[j - 1] * lanczos_mat[j - 1] if j > 0 else 0)
        beta[j] = np.linalg.norm(w_j)
        if beta[j] < 100 * len(vec) * np.finfo(float).eps:
            # Terminate early if the next vector is (numerically) zero.
            lanczos_iterations = j + 1
            return (alpha[:lanczos_iterations], beta[: lanczos_iterations - 1], lanczos_mat[:lanczos_iterations, :].T)
        lanczos_mat[j + 1] = w_j / beta[j]

    # Complete final iteration
    j = lanczos_iterations - 1
    w_j = matrix_free_operator(lanczos_mat[j])
    alpha[j] = np.vdot(w_j, lanczos_mat[j]).real
    return (alpha, beta, lanczos_mat.T)


def expm_krylov(
    matrix_free_operator: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
    vec: NDArray[np.complex128],
    dt: float,
    lanczos_iterations: int,
) -> NDArray[np.complex128]:
    """Compute the Krylov subspace approximation of the matrix exponential applied to a vector.

    This function approximates exp(-1j * dt * A) * v by projecting the action of the matrix exponential
    onto a Krylov subspace generated by the Lanczos iteration. The method is based on the approach
    described by Hochbruck and Lubich.

    Args:
        matrix_free_operator (Callable[[NDArray[np.complex128]], NDArray[np.complex128]]):
            A function implementing the matrix-free application of the linear operator A.
        vec (NDArray[np.complex128]):
            The input vector to which the matrix exponential is applied.
        dt (float):
            The time step (or scalar multiplier) in the exponential.
        lanczos_iterations (int):
            The number of Lanczos iterations (and the dimension of the Krylov subspace) to use.

    Returns:
        NDArray[np.complex128]:
            The approximate result of applying exp(-1j * dt * A) to vec.
    """
    alpha, beta, lanczos_mat = lanczos_iteration(matrix_free_operator, vec, lanczos_iterations)
    try:
        w_hess, u_hess = eigh_tridiagonal(alpha, beta, lapack_driver="stemr")
    except np.linalg.LinAlgError:
        # Fallback to stable but potentially slower solver
        w_hess, u_hess = eigh_tridiagonal(alpha, beta, lapack_driver="stebz")
    # Construct the approximation: scale the exponential of the eigenvalues by the norm of v,
    # and project back to the full space via the Lanczos basis V.
    return lanczos_mat @ (u_hess @ (np.linalg.norm(vec) * np.exp(-1j * dt * w_hess) * u_hess[0]))
