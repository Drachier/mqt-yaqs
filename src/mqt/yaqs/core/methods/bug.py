# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Implements the Basis-Update and Galerkin Method (BUG) for MPS.

Refer to Ceruti et al. (2021) doi:10.1137/22M1473790 for details of the method
for TTN.
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import rq, qr

from ..data_structures.simulation_parameters import StrongSimParams, WeakSimParams
from .decompositions import left_qr, right_qr
from .tdvp import update_left_environment, update_right_environment, update_site
from .matrix_exponential import expm_krylov

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..data_structures.networks import MPO, MPS
    from ..data_structures.simulation_parameters import PhysicsSimParams


def prepare_canonical_site_tensors(
    state: MPS, mpo: MPO
) -> tuple[list[NDArray[np.complex128]], list[NDArray[np.complex128]]]:
    """We need to get the original tensor when every site is the canonical form.

    Assumes the MPS is in the left-canonical form.

    Args:
        state: The MPS.
        mpo: The MPO.

    Returns:
        canon_tensors: The list of the canonical site tensors.
        left_blocks: The list of the left environments.

    """
    # This will merely do a shallow copy of the MPS.
    canon_tensors = copy(state.tensors)
    left_end_dimension = state.tensors[0].shape[1]
    left_blocks = [np.eye(left_end_dimension).reshape(left_end_dimension, 1, left_end_dimension)]
    for i, old_local_tensor in enumerate(canon_tensors[1:], start=1):
        left_tensor = canon_tensors[i - 1]
        left_q, left_r = right_qr(left_tensor)
        # Legs of right_r: (new, old_right)
        local_tensor = np.tensordot(left_r, old_local_tensor, axes=(1, 1))
        # Leg order of local_tensor: (left, phys, right)
        local_tensor = local_tensor.transpose(1, 0, 2)
        # Correct leg order: (phys, left, right) and orth center
        canon_tensors[i] = local_tensor
        new_env = update_left_environment(left_q, left_q, mpo.tensors[i - 1], left_blocks[i - 1])
        left_blocks.append(new_env)
    return canon_tensors, left_blocks


def choose_stack_tensor(
    site: int, canon_center_tensors: list[NDArray[np.complex128]], state: MPS
) -> NDArray[np.complex128]:
    """Return the non-update tensor that should be used for the stacking step.

    If the site is the last one and thus the leaf site, we need to choose the
    MPS tensor, when the MPS was in left-canonical form. Otherwise, we choose
    the MPS tensor, when the local site was the orthognality center.

    Args:
        site: The site to be updated.
        canon_center_tensors: The canonical site tensors.
        state: The MPS.

    Returns:
        NDArray[np.complex128]: The tensor to be stacked.

    """
    if site == state.length - 1:  # noqa: SIM108
        # This is the only leaf case.
        old_stack_tensor = state.tensors[site]
    else:
        old_stack_tensor = canon_center_tensors[site]
    return old_stack_tensor


def find_new_q(
    old_stack_tensor: NDArray[np.complex128], updated_tensor: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Finds the new Q tensor after the update with enlarged left virtual leg.

    Args:
        old_stack_tensor: The tensor to be stacked with the updated tensor.
        updated_tensor: The tensor after the update.

    Returns:
        new_q: The new Q tensor with MPS leg order (phys, left, right).

    """
    stacked_tensor = np.concatenate((old_stack_tensor, updated_tensor), axis=1)
    new_q, _ = left_qr(stacked_tensor)
    return new_q


def build_basis_change_tensor(
    old_q: NDArray[np.complex128], new_q: NDArray[np.complex128], old_m: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Build a new basis change tensor M.

    Args:
        old_q: The old tensor of the site, when the MPS was in left-canonical
            form. The leg order is (phys, left, right).
        new_q: The extended local base tensor after the update. Same leg order
            as an MPS tensor. The leg order is (phys, left, right).
        old_m: The basis change matrix of the site to the right. The leg order
            is (old,new).

    Returns:
        new_m: The basis change tensor M. The leg order is (old,new).

    """
    new_m = np.tensordot(old_q, old_m, axes=(2, 0))
    return np.tensordot(new_m, new_q.conj(), axes=([0, 2], [0, 2]))

def project_phys(
    left_block: NDArray[np.complex128],
    right_block: NDArray[np.complex128],
    mpo_tensor: NDArray[np.complex128],
    site_tensor: NDArray[np.complex128],
    phys_tensor: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """
    Build the projector for the tensor on the physical leg.

    Args:
        left_block: The left environment of the site.
        right_block: The right environment of the site.
        mpo_tensor: The MPO tensor corresponding to the site.
        site_tensor: The local physical tensor of the MPS.
        phys_tensor: The physical tensor of the MPS.
    
    Returns:
        projected_tensor: The projected tensor.
    """
    full_site_tensor = np.tensordot(phys_tensor,
                                    site_tensor,
                                    axes=(1,0))
    tensor = np.tensordot(full_site_tensor,
                          right_block,
                          axes=(2,0))
    tensor = np.tensordot(tensor,
                          mpo_tensor,
                          axes=([0,2],[1,3]))
    tensor = np.tensordot(tensor,
                          site_tensor.conj(),
                          axes=(1,2))
    tensor = np.tensordot(tensor,
                          left_block,
                          axes=([0,2,4],[0,1,2]))
    return tensor

def update_phys_tensor(
    phys_tensor: NDArray[np.complex128],
    site_tensor: NDArray[np.complex128],
    mpo_tensor: NDArray[np.complex128],
    left_block: NDArray[np.complex128],
    right_block: NDArray[np.complex128],
    dt: float,
    numiter_lanczos: int
) -> NDArray[np.complex128]:
    """
    Update the physical tensor of the MPS.

    Args:
        phys_tensor: The physical tensor of the MPS.
        site_tensor: The local physical tensor of the MPS.
        mpo_tensor: The MPO tensor corresponding to the site.
        left_block: The left environment of the site.
        right_block: The right environment of the site.
        dt: Time step for the simulation.
        numiter_lanczos: Number of Lanczos iterations.
    
    Returns:
        updated_tensor: The updated physical tensor.
    """
    phys_flat = phys_tensor.reshape(-1)
    evolved_tensor_flat = expm_krylov(
        lambda x: project_phys(left_block, right_block, mpo_tensor, site_tensor, x.reshape(phys_tensor.shape)).reshape(-1),
        phys_flat,
        dt,
        numiter_lanczos
    )
    return evolved_tensor_flat.reshape(phys_tensor.shape)

def update_local_phys_tensor(
        site_tensor: NDArray[np.complex128],
        mpo_tensor: NDArray[np.complex128],
        left_block: NDArray[np.complex128],
        right_block: NDArray[np.complex128],
        sim_params: PhysicsSimParams | WeakSimParams | StrongSimParams,
        numiter_lanczos: int
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Update the local physical tensor of the MPS.

    Args:
        site_tensor: The local physical tensor of the MPS.
        mpo_tensor: The MPO tensor corresponding to the site.
        left_block: The left environment of the site.
        right_block: The right environment of the site.
        sim_params: Simulation parameters.
        numiter_lanczos: Number of Lanczos iterations.
    
    Returns:
        updated_tensor: The updated local physical tensor.
        m_tensor: The basis change tensor M.
        block: The evironment with the physical leg. Basically a basis change of the MPO tensor.
        old_tensor: The old tensor of the site, which is the orthogonality center, but with a physical tensor factored out.
    """
    # First get the physical tensor
    # QR upwards
    temp_tens = site_tensor.reshape((site_tensor.shape[0],-1)) # (phys, left*right)
    phys_tensor, site_tensor_canon = rq(temp_tens)
    # Phys tensor shape = (outer, inner), site_tensor shape = (inner, left*right)
    site_tensor_canon = site_tensor_canon.reshape((site_tensor.shape[0],-1,site_tensor.shape[2])) # (inner, left, right)
    new_phys_tensor = update_phys_tensor(phys_tensor, site_tensor_canon, mpo_tensor, left_block, right_block,
                                         sim_params.dt, numiter_lanczos) # (outer, inner)
    # get the old basis tensor via QR downwards
    phys_tensor, r = qr(phys_tensor)
    stack_tensor = np.concatenate((phys_tensor, new_phys_tensor), axis=1) # (outer, 2*inner)
    # QR downwards
    new_phys_tensor, _ = qr(stack_tensor)
    # Now we have the new physical tensor and get the basis change tensor
    m_tensor = np.tensordot(new_phys_tensor.conj(),
                            phys_tensor,
                            axes=(0,0)) # (inner_new, inner_old)
    # Now we need to get the new environment with the physical leg
    block = np.tensordot(new_phys_tensor.conj(),
                         mpo_tensor,
                         axes=(0,0)) # (inner_conj, in, left, right)
    block = np.tensordot(new_phys_tensor,
                         block,
                         axes=(0,1)) #(inner, inner_conj, left, right)
    block = block.transpose(1,0,2,3) # (inner_conj, inner, left, right)
    # Build old tensor
    old_tensor = np.tensordot(r,
                              site_tensor_canon,
                              axes=(1,0)) # (inner, left*right)
    return new_phys_tensor, m_tensor, block, old_tensor

def local_update(
    state: MPS,
    mpo: MPO,
    left_blocks: list[NDArray[np.complex128]],
    right_block: NDArray[np.complex128],
    canon_center_tensors: list[NDArray[np.complex128]],
    site: int,
    right_m_block: NDArray[np.complex128],
    right_old_block: NDArray[np.complex128],
    sim_params: PhysicsSimParams | WeakSimParams | StrongSimParams,
    numiter_lanczos: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Single Site bug algorithm update.

    Updates a single site of the MPS.

    Args:
        state: The MPS.
        mpo: The MPO.
        left_blocks: The left environments.
        right_block: The right environment.
        canon_center_tensors: The canonical site tensors.
        site: The site to be updated.
        right_m_block: The basis update matrix of the site to the right.
        right_old_block: The right environment of the site to be updated with the old MPS tensors.
        sim_params: Simulation parameters.
        numiter_lanczos: Number of Lanczos iterations.

    Returns:
        basis_change_m: The basis update matrix of this site.
        new_right_block: The right environment of this site.
        new_old_block: The right environment of this site with the updated MPS
    """
    old_tensor = canon_center_tensors[site]
    # Update the local physical tensor of the MPS.
    new_phys_tensor, phys_m, phys_block, old_tensor = update_local_phys_tensor(
                                                                old_tensor,
                                                                mpo.tensors[site],
                                                                left_blocks[site],
                                                                right_old_block,
                                                                sim_params,
                                                                numiter_lanczos)
    # Pull in the basis change tensor
    old_tensor = np.tensordot(phys_m, old_tensor, axes=(1,0)) # (new, left, right)
    updated_tensor = update_site(
        left_blocks[site], right_block, phys_block, old_tensor, sim_params.dt, numiter_lanczos
    ) # Remeber phys_block is the basis changed MPO tensor
    old_stack_tensor = choose_stack_tensor(site, canon_center_tensors, state)
    new_q = find_new_q(old_stack_tensor, updated_tensor)
    old_q = state.tensors[site]
    basis_change_m = build_basis_change_tensor(old_q, new_q, right_m_block)
    # Build new old block
    old_site_tensor = state.tensors[site]
    new_old_block = update_right_environment(old_site_tensor, old_site_tensor, mpo.tensors[site], right_old_block)
    # Build new site tensor
    new_site_tensor = np.tensordot(new_phys_tensor, new_q, axes=(1, 0))
    state.tensors[site] = new_site_tensor
    canon_center_tensors[site - 1] = np.tensordot(canon_center_tensors[site - 1], basis_change_m, axes=(2, 0))
    new_right_block = update_right_environment(new_q, new_q, mpo.tensors[site], right_block)
    return basis_change_m, new_right_block, new_old_block


def bug(
    state: MPS, mpo: MPO, sim_params: PhysicsSimParams | WeakSimParams | StrongSimParams, numiter_lanczos: int = 25
) -> None:
    """Performs the Basis-Update and Galerkin Method for an MPS.

    The state is updated in place.

    Args:
        mpo: Hamiltonian represented as an MPO.
        state: The initial state represented as an MPS.
        sim_params: Simulation parameters containing time step 'dt' and SVD
            threshold.
        numiter_lanczos: Number of Lanczos iterations for each local update.

    Raises:
        ValueError: If the state and Hamiltonian have different numbers of
            sites.

    """
    num_sites = mpo.length
    if num_sites != state.length:
        msg = "State and Hamiltonian must have the same number of sites"
        raise ValueError(msg)

    if isinstance(sim_params, (WeakSimParams, StrongSimParams)):
        sim_params.dt = 1

    canon_center_tensors, left_envs = prepare_canonical_site_tensors(state, mpo)
    right_end_dimension = state.tensors[-1].shape[2]
    right_block = np.eye(right_end_dimension).reshape(right_end_dimension, 1, right_end_dimension)
    right_old_block = np.eye(right_end_dimension).reshape(right_end_dimension, 1, right_end_dimension)
    right_m_block = np.eye(right_end_dimension)
    # Sweep from right to left.
    for site in range(num_sites - 1, 0, -1):
        right_m_block, right_block, right_old_block = local_update(
            state, mpo, left_envs, right_block, canon_center_tensors, site, right_m_block, right_old_block, sim_params, numiter_lanczos
        )
    # Update the first site.
    updated_tensor = update_site(
        left_envs[0], right_block, mpo.tensors[0], canon_center_tensors[0], sim_params.dt, numiter_lanczos
    )
    state.tensors[0] = updated_tensor
    # Truncation
    state.truncate(sim_params.threshold, sim_params.max_bond_dim)
