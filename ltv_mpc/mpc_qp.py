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

from .exceptions import ProblemDefinitionError


class MPCQP:
    r"""MPC problem represented as a quadratic program.

    This class further stores intermediate matrices used to recompute cost and
    linear inequality vectors.
    """

    def __init__(self, problem) -> None:
        input_dim = problem.input_dim
        state_dim = problem.state_dim
        stacked_input_dim = problem.input_dim * problem.nb_timesteps
        if problem.initial_state is None:
            raise ProblemDefinitionError("initial state is undefined")
        initial_state: np.ndarray = problem.initial_state

        phi = np.eye(state_dim)
        psi = np.zeros((state_dim, stacked_input_dim))
        G_list, h_list = [], []
        phi_list, psi_list = [], []
        for k in range(problem.nb_timesteps):
            # Loop invariant: x == psi * U + phi * x_init
            phi_list.append(phi)
            psi_list.append(psi)
            A_k = problem.get_transition_state_matrix(k)
            B_k = problem.get_transition_input_matrix(k)
            C_k = problem.get_ineq_state_matrix(k)
            D_k = problem.get_ineq_input_matrix(k)
            e_k = problem.get_ineq_vector(k)
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

        Phi = np.vstack(phi_list)
        Psi = np.vstack(psi_list)
        P: np.ndarray = problem.stage_input_cost_weight * np.eye(
            stacked_input_dim,
        )
        q: np.ndarray = np.zeros(stacked_input_dim)
        if problem.has_terminal_cost:
            if problem.goal_state is None:
                raise ProblemDefinitionError("goal state is undefined")
            P += problem.terminal_cost_weight * np.dot(psi.T, psi)
        if problem.has_stage_state_cost:
            if problem.target_states is None:
                raise ProblemDefinitionError(
                    "reference trajectory is undefined"
                )
            P += problem.stage_state_cost_weight * np.dot(Psi.T, Psi)

        self.G: np.ndarray = np.vstack(G_list)
        self.P = P
        self.Phi = Phi
        self.Psi = Psi
        self.h: np.ndarray = np.hstack(h_list)
        self.phi_last = phi
        self.psi_last = psi
        self.q = q  # initialized below
        #
        self.recompute_cost_vector(problem)

    def update_cost_vector(self, problem) -> None:
        if problem.initial_state is None:
            raise ProblemDefinitionError("initial state is undefined")
        initial_state = problem.initial
        self.q[:] = 0.0
        if problem.has_terminal_cost:
            if problem.goal_state is None:
                raise ProblemDefinitionError("goal state is undefined")
            c = np.dot(self.phi_last, initial_state) - problem.goal_state
            self.q += problem.terminal_cost_weight * np.dot(c.T, self.psi_last)
        if problem.has_stage_state_cost:
            if problem.target_states is None:
                raise ProblemDefinitionError(
                    "reference trajectory is undefined"
                )
            c = np.dot(self.Phi, problem.initial_state) - problem.target_states
            self.q += problem.stage_state_cost_weight * np.dot(c.T, self.Psi)

    def update_constraint_vector(self, problem) -> None:
        raise NotImplementedError(
            "Time-varying constraints are handled cold-start for now"
        )
