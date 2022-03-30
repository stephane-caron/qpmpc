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

import numpy as np

from scipy.sparse import csc_matrix

from .problem import Problem
from .mpcqp import MPCQP


def build_mpcqp(problem: Problem, sparse: bool = False) -> MPCQP:
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
    return MPCQP(cost_matrix=P, cost_vector=q, ineq_matrix=G, ineq_vector=h)
