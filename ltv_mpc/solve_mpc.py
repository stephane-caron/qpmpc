#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Stéphane Caron
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from scipy.sparse import csc_matrix
from qpsolvers import solve_qp

from .problem import Problem
from .solution import Solution


@dataclass
class QuadraticProgram:

    """
    Quadratic program (QP) with inequality constraints.
    """

    cost_matrix: Union[np.ndarray, csc_matrix]
    cost_vector: np.ndarray
    ineq_matrix: Union[np.ndarray, csc_matrix]
    ineq_vector: np.ndarray


def build_qp(problem: Problem, sparse: bool = False) -> QuadraticProgram:
    """
    Build the quadratic program corresponding to an LTV-MPC problem.

    Args:
        problem: Model predictive control problem.
        sparse: Whether to use sparse or dense matrices in the output quadratic
            program. Enable it if you are calling a sparse solver afterwards.

    Returns:
        Quadratic program representing the input problem.

    Notes:
        In numerical analysis, there are three classes of methods to solve
        boundary value problems: single shooting, multiple shooting and
        collocation. The QP built by this function implements a `single
        shooting method <https://en.wikipedia.org/wiki/Shooting_method>`_.
    """
    input_dim = problem.input_dim
    state_dim = problem.state_dim
    stacked_input_dim = problem.input_dim * problem.nb_timesteps

    phi = np.eye(state_dim)
    psi = np.zeros((state_dim, stacked_input_dim))
    G_list, h_list = [], []
    phi_list, psi_list = [], []
    for k in range(problem.nb_timesteps):
        # Loop invariant: x == psi * U + phi * x_init
        if problem.stage_state_cost_weight is not None:
            phi_list.append(phi)
            psi_list.append(psi)
        A = problem.get_transition_state_matrix(k)
        B = problem.get_transition_input_matrix(k)
        C = problem.get_ineq_state_matrix(k)
        D = problem.get_ineq_input_matrix(k)
        e = problem.get_ineq_vector(k)
        G = np.zeros((e.shape[0], stacked_input_dim))
        h = e if C is None else e - np.dot(C.dot(phi), problem.initial_state)
        input_slice = slice(k * input_dim, (k + 1) * input_dim)
        if D is not None:
            # we rely on G == 0 to avoid a slower +=
            G[:, input_slice] = D
        if C is not None:
            G += C.dot(psi)
        if k == 0 and D is None:  # corner case, input has no effect
            assert np.all(h >= 0.0)
        else:  # regular case, G is non-zero
            G_list.append(G)
            h_list.append(h)
        phi = A.dot(phi)
        psi = A.dot(psi)
        psi[:, input_slice] = B

    P = problem.stage_input_cost_weight * np.eye(stacked_input_dim)
    q = np.zeros(stacked_input_dim)
    if (
        problem.terminal_cost_weight is not None
        and problem.terminal_cost_weight > 1e-10
    ):
        c = np.dot(phi, problem.initial_state) - problem.goal_state
        P += problem.terminal_cost_weight * np.dot(psi.T, psi)
        q += problem.terminal_cost_weight * np.dot(c.T, psi)
    if (
        problem.stage_state_cost_weight is not None
        and problem.stage_state_cost_weight > 1e-10
    ):
        Phi = np.vstack(phi_list)
        Psi = np.vstack(psi_list)
        X_goal = np.hstack([problem.goal_state] * problem.nb_timesteps)
        c = np.dot(Phi, problem.initial_state) - X_goal
        P += problem.stage_state_cost_weight * np.dot(Psi.T, Psi)
        q += problem.stage_state_cost_weight * np.dot(c.T, Psi)

    G = np.vstack(G_list)
    h = np.hstack(h_list)
    if sparse:
        P = csc_matrix(P)
        G = csc_matrix(G)
    return QuadraticProgram(
        cost_matrix=P, cost_vector=q, ineq_matrix=G, ineq_vector=h
    )


def solve_mpc(
    problem: Problem,
    sparse: bool = False,
    solver: Optional[str] = None,
    **kwargs
) -> Solution:
    """
    Solve a linear time-invariant model predictive control problem.

    Args:
        problem: Model predictive control problem to solve.
        solver: Quadratic programming solver to use, to choose in
            :data:`qpsolvers.available_solvers`. Both "quadprog" and "osqp"
            tend to perform well on model predictive control problems. See for
            instance `this benchmark
            <https://github.com/stephane-caron/qpsolvers#benchmark>`__.
        sparse: Whether to use sparse or dense matrices in the output quadratic
            program. Enable it if the QP solver is sparse (e.g. OSQP).

    Returns:
        Solution to the problem, if found.

    Note:
        Keyword arguments are passed to the QP solver via the `solve_qp`_
        function. In particular, the ``solver`` string can be set to select a
        different QP solver.

    .. _solve_qp:
        https://scaron.info/doc/qpsolvers/quadratic-programming.html#qpsolvers.solve_qp
    """
    qp = build_qp(problem, sparse=sparse)
    U = solve_qp(
        qp.cost_matrix,
        qp.cost_vector,
        qp.ineq_matrix,
        qp.ineq_vector,
        solver=solver,
        **kwargs
    )
    U = U.reshape((problem.nb_timesteps, problem.input_dim))
    return Solution(problem, U)
