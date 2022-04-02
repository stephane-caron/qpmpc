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

from typing import List, Optional, Union

import numpy as np


class Problem:

    """
    Linear time-variant model predictive control problem.

    The discretized dynamics of a linear system are described by:

    .. math::

        x_{k+1} = A x_k + B u_k

    where :math:`x` is assumed to be the first-order state of a configuration
    variable :math:`p`, i.e., it stacks both the position :math:`p` and its
    time-derivative :math:`\\dot{p}`. Meanwhile, the system is linearly
    constrained by:

    .. math::

        x_0 & = x_\\mathrm{init} \\\\
        \\forall k, \\ C_k x_k + D_k u_k & \\leq e_k \\\\

    The output control law minimizes a weighted combination of two types of
    costs:

    - Terminal state error
        :math:`\\|x_\\mathrm{nb\\_steps} - x_\\mathrm{goal}\\|^2`
        with weight :math:`w_{xt}`.
    - Cumulated state error:
        :math:`\\sum_k \\|x_k - x_\\mathrm{goal}\\|^2`
        with weight :math:`w_{xc}`.
    - Cumulated control costs:
        :math:`\\sum_k \\|u_k\\|^2`
        with weight :math:`w_{u}`.

    Attributes:
        transition_state_matrix: State linear dynamics matrix.
        transition_input_matrix: Control linear dynamics matrix.
        ineq_state_matrix : Constraint matrix on state variables. When this
            argument is an array, the same matrix `C` is applied at each step
            `k`. When it is ``None``, the null matrix is applied.
        ineq_input_matrix : Constraint matrix on control variables. When this
            argument is an array, the same matrix `D` is applied at each step
            `k`. When it is ``None``, the null matrix is applied.
        ineq_vector : Constraint vector. When this argument is an array, the
            same vector `e` is applied at each step `k`.
        initial_state: Initial state as stacked position and velocity.
        goal_state: Goal state as stacked position and velocity.
        nb_timesteps: Number of discretization steps in the preview window.
        terminal_cost_weight: Weight on terminal state cost, or ``None`` to
            disable.
        stage_state_cost_weight: Weight on cumulated state costs, or ``None``
            to disable (default).
        stage_input_cost_weight: Weight on cumulated control costs.
        input_dim: Dimension of an input vector.
        state_dim: Dimension of a state vector.
    """

    transition_state_matrix: Union[np.ndarray, List[np.ndarray]]
    transition_input_matrix: Union[np.ndarray, List[np.ndarray]]
    ineq_state_matrix: Union[None, np.ndarray, List[np.ndarray]]
    ineq_input_matrix: Union[None, np.ndarray, List[np.ndarray]]
    ineq_vector: Union[np.ndarray, List[np.ndarray]]
    initial_state: np.ndarray
    goal_state: np.ndarray
    nb_timesteps: int
    terminal_cost_weight: Optional[float]
    stage_state_cost_weight: Optional[float]
    stage_input_cost_weight: float

    input_dim: int
    state_dim: int

    def __init__(
        self,
        transition_state_matrix: Union[np.ndarray, List[np.ndarray]],
        transition_input_matrix: Union[np.ndarray, List[np.ndarray]],
        ineq_state_matrix: Union[None, np.ndarray, List[np.ndarray]],
        ineq_input_matrix: Union[None, np.ndarray, List[np.ndarray]],
        ineq_vector: Union[np.ndarray, List[np.ndarray]],
        initial_state: np.ndarray,
        goal_state: np.ndarray,
        nb_timesteps: int,
        terminal_cost_weight: Optional[float],
        stage_state_cost_weight: Optional[float],
        stage_input_cost_weight: float,
    ) -> None:
        assert (
            ineq_state_matrix is not None or ineq_input_matrix is not None
        ), "use LQR for unconstrained case"
        assert (
            stage_input_cost_weight > 0.0
        ), "non-negative control weight needed for regularization"
        assert (
            terminal_cost_weight is not None
            or stage_state_cost_weight is not None
        ), "set either wxt or wxc"
        self.transition_state_matrix = transition_state_matrix
        self.transition_input_matrix = transition_input_matrix
        self.ineq_state_matrix = ineq_state_matrix
        self.ineq_input_matrix = ineq_input_matrix
        self.ineq_vector = ineq_vector
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.nb_timesteps = nb_timesteps
        self.terminal_cost_weight = terminal_cost_weight
        self.stage_state_cost_weight = stage_state_cost_weight
        self.stage_input_cost_weight = stage_input_cost_weight
        if isinstance(transition_input_matrix, np.ndarray):
            self.input_dim = transition_input_matrix.shape[1]
        else:  # isinstance(transition_input_matrix, List[np.ndarray])
            self.input_dim = transition_input_matrix[0].shape[1]
        if isinstance(transition_state_matrix, np.ndarray):
            self.state_dim = transition_state_matrix.shape[1]
        else:  # isinstance(transition_state_matrix, List[np.ndarray])
            self.state_dim = transition_state_matrix[0].shape[1]

    def get_transition_state_matrix(self, k):
        return (
            self.transition_state_matrix[k]
            if isinstance(self.transition_state_matrix, list)
            else self.transition_state_matrix
        )

    def get_transition_input_matrix(self, k):
        return (
            self.transition_input_matrix[k]
            if isinstance(self.transition_input_matrix, list)
            else self.transition_input_matrix
        )

    def get_ineq_state_matrix(self, k):
        return (
            self.ineq_state_matrix[k]
            if isinstance(self.ineq_state_matrix, list)
            else self.ineq_state_matrix
        )

    def get_ineq_input_matrix(self, k):
        return (
            self.ineq_input_matrix[k]
            if isinstance(self.ineq_input_matrix, list)
            else self.ineq_input_matrix
        )

    def get_ineq_vector(self, k):
        return (
            self.ineq_vector[k]
            if isinstance(self.ineq_vector, list)
            else self.ineq_vector
        )
