#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria
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

"""MPC problem represented as a quadratic program."""

from logging import warn

import numpy as np
import qpsolvers
from scipy.sparse import csc_matrix

from .exceptions import ProblemDefinitionError
from .mpc_problem import MPCProblem


class MPCQP:
    r"""MPC problem represented as a quadratic program.

    This class further stores intermediate matrices used to recompute cost and
    linear inequality vectors.
    """

    G: np.ndarray
    P: np.ndarray
    Phi: np.ndarray
    Psi: np.ndarray
    h: np.ndarray
    phi_last: np.ndarray
    psi_last: np.ndarray
    q: np.ndarray

    def __init__(self, mpc_problem: MPCProblem, sparse: bool = False) -> None:
        """Create a new QP representation.

        Args:
            mpc_problem: Model predictive control problem to cast as a QP.
            sparse: If set, use sparse matrix representation.
        """
        input_dim = mpc_problem.input_dim
        state_dim = mpc_problem.state_dim
        stacked_input_dim = mpc_problem.input_dim * mpc_problem.nb_timesteps
        if mpc_problem.initial_state is None:
            raise ProblemDefinitionError("initial state is undefined")
        initial_state: np.ndarray = mpc_problem.initial_state

        phi = np.eye(state_dim)
        psi = np.zeros((state_dim, stacked_input_dim))
        G_list, h_list = [], []
        phi_list, psi_list = [], []
        for k in range(mpc_problem.nb_timesteps):
            # Loop invariant: x == psi * U + phi * x_init
            phi_list.append(phi)
            psi_list.append(psi)
            A_k = mpc_problem.get_transition_state_matrix(k)
            B_k = mpc_problem.get_transition_input_matrix(k)
            C_k = mpc_problem.get_ineq_state_matrix(k)
            D_k = mpc_problem.get_ineq_input_matrix(k)
            e_k = mpc_problem.get_ineq_vector(k)
            G_k = np.zeros((e_k.shape[0], stacked_input_dim))
            h_k = (
                e_k
                if C_k is None
                else e_k - np.dot(C_k.dot(phi), initial_state)
            )
            input_slice = slice(k * input_dim, (k + 1) * input_dim)
            if D_k is not None:
                # we rely on G == 0 to avoid a slower +=
                G_k[:, input_slice] = D_k
            if C_k is not None:
                G_k += C_k.dot(psi)
            if k == 0 and D_k is None and np.any(h_k < 0.0):
                # in this case, the initial state constraint is violated and
                # cannot be compensated by any input (D_k is None)
                warn(
                    "initial state is unfeasible: "
                    f"G_0 * x <= h_0 with G_0 == 0 and min(h_0) == {min(h_k)}"
                )
            G_list.append(G_k)
            h_list.append(h_k)
            phi = A_k.dot(phi)
            psi = A_k.dot(psi)
            psi[:, input_slice] = B_k
        G: np.ndarray = np.vstack(G_list)
        h: np.ndarray = np.hstack(h_list)
        Phi = np.vstack(phi_list)
        Psi = np.vstack(psi_list)

        P: np.ndarray = mpc_problem.stage_input_cost_weight * np.eye(
            stacked_input_dim,
        )
        if mpc_problem.terminal_cost_weight is not None:
            P += mpc_problem.terminal_cost_weight * np.dot(psi.T, psi)
        if mpc_problem.stage_state_cost_weight is not None:
            P += mpc_problem.stage_state_cost_weight * np.dot(Psi.T, Psi)
        q: np.ndarray = np.zeros(stacked_input_dim)

        self.G = csc_matrix(G) if sparse else G
        self.P = csc_matrix(P) if sparse else P
        self.Phi = Phi
        self.Psi = Psi
        self.h = h
        self.phi_last = phi
        self.psi_last = psi
        self.q = q  # initialized below
        #
        try:
            self.update_cost_vector(mpc_problem)
        except ProblemDefinitionError:
            pass

    @property
    def problem(self) -> qpsolvers.Problem:
        """Get quadratic program to call a QP solver."""
        return qpsolvers.Problem(self.P, self.q, self.G, self.h)

    def update_cost_vector(self, mpc_problem: MPCProblem) -> None:
        """Update the gradient vector in the cost function.

        Args:
            mpc_problem: New model predictive control problem. It should have
                the same structure as the one used to initialize the MPCQP.
        """
        if mpc_problem.initial_state is None:
            raise ProblemDefinitionError("initial state is undefined")
        initial_state = mpc_problem.initial_state
        self.q[:] = 0.0
        if mpc_problem.has_terminal_cost:
            c = np.dot(self.phi_last, initial_state) - mpc_problem.goal_state
            self.q += mpc_problem.terminal_cost_weight * np.dot(
                c.T, self.psi_last
            )
        if mpc_problem.has_stage_state_cost:
            c = np.dot(self.Phi, initial_state) - mpc_problem.target_states
            self.q += mpc_problem.stage_state_cost_weight * np.dot(
                c.T, self.Psi
            )

    def update_constraint_vector(self, mpc_problem: MPCProblem) -> None:
        """Update the inequality constraint vector.

        Args:
            mpc_problem: New model predictive control problem. It should have
                the same structure as the one used to initialize the MPCQP.
        """
        raise NotImplementedError(
            "Time-varying constraints are handled cold-start for now"
        )
