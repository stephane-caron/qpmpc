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

"""Define model predictive control problems."""

from typing import List, Optional, Union

import numpy as np

from .exceptions import ProblemDefinitionError, StateError


class MPCProblem:
    r"""Linear time-variant model predictive control problem.

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
        goal_state: Goal state as stacked position and velocity.
        ineq_input_matrix: Constraint matrix on control variables. When this
            argument is an array, the same matrix `D` is applied at each step
            `k`. When it is ``None``, the null matrix is applied.
        ineq_state_matrix: Constraint matrix on state variables. When this
            argument is an array, the same matrix `C` is applied at each step
            `k`. When it is ``None``, the null matrix is applied.
        ineq_vector: Constraint vector. When this argument is an array, the
            same vector `e` is applied at each step `k`.
        initial_state: Initial state as stacked position and velocity.
        input_dim: Dimension of an input vector.
        nb_timesteps: Number of discretization steps in the preview window.
        stage_input_cost_weight: Weight on cumulated control costs.
        stage_state_cost_weight: Weight on cumulated state costs, or ``None``
            to disable (default).
        state_dim: Dimension of a state vector.
        terminal_cost_weight: Weight on terminal state cost, or ``None`` to
            disable.
        transition_input_matrix: Control linear dynamics matrix.
        transition_state_matrix: State linear dynamics matrix.
    """

    goal_state: Optional[np.ndarray]
    ineq_input_matrix: Union[None, np.ndarray, List[np.ndarray]]
    ineq_state_matrix: Union[None, np.ndarray, List[np.ndarray]]
    ineq_vector: Union[np.ndarray, List[np.ndarray]]
    initial_state: Optional[np.ndarray]
    input_dim: int
    nb_timesteps: int
    stage_input_cost_weight: float
    stage_state_cost_weight: Optional[float]
    state_dim: int
    target_states: Optional[np.ndarray]
    terminal_cost_weight: Optional[float]
    transition_input_matrix: Union[np.ndarray, List[np.ndarray]]
    transition_state_matrix: Union[np.ndarray, List[np.ndarray]]

    def __init__(
        self,
        transition_state_matrix: Union[np.ndarray, List[np.ndarray]],
        transition_input_matrix: Union[np.ndarray, List[np.ndarray]],
        ineq_state_matrix: Union[None, np.ndarray, List[np.ndarray]],
        ineq_input_matrix: Union[None, np.ndarray, List[np.ndarray]],
        ineq_vector: Union[np.ndarray, List[np.ndarray]],
        nb_timesteps: int,
        terminal_cost_weight: Optional[float],
        stage_state_cost_weight: Optional[float],
        stage_input_cost_weight: float,
        initial_state: Optional[np.ndarray] = None,
        goal_state: Optional[np.ndarray] = None,
        target_states: Optional[np.ndarray] = None,
    ) -> None:
        """Start a new model predictive control problem."""
        if stage_input_cost_weight <= 0.0:
            raise ProblemDefinitionError(
                "stage non-negative control weight needed for regularization"
            )
        if terminal_cost_weight is None and stage_state_cost_weight is None:
            raise ProblemDefinitionError(
                "either terminal or stage state cost should be set"
            )
        input_dim = (
            transition_input_matrix.shape[1]
            if isinstance(transition_input_matrix, np.ndarray)
            else transition_input_matrix[0].shape[1]
        )
        state_dim = (
            transition_state_matrix.shape[1]
            if isinstance(transition_state_matrix, np.ndarray)
            else transition_state_matrix[0].shape[1]
        )
        self.goal_state = None  # initialized below
        self.ineq_input_matrix = ineq_input_matrix
        self.ineq_state_matrix = ineq_state_matrix
        self.ineq_vector = ineq_vector
        self.initial_state = None  # initialized below
        self.input_dim = input_dim
        self.nb_timesteps = nb_timesteps
        self.target_states = None  # initialized below
        self.stage_input_cost_weight = stage_input_cost_weight
        self.stage_state_cost_weight = stage_state_cost_weight
        self.state_dim = state_dim
        self.terminal_cost_weight = terminal_cost_weight
        self.transition_input_matrix = transition_input_matrix
        self.transition_state_matrix = transition_state_matrix
        if goal_state is not None:
            self.update_goal_state(goal_state)
        if initial_state is not None:
            self.update_initial_state(initial_state)

    @property
    def has_terminal_cost(self) -> bool:
        """Check whether the problem has a terminal cost."""
        cost_is_set = (
            self.terminal_cost_weight is not None
            and self.terminal_cost_weight > 1e-10
        )
        if cost_is_set and self.goal_state is None:
            raise ProblemDefinitionError(
                "MPC problem has terminal cost "
                "but the goal state is undefined"
            )
        return cost_is_set

    @property
    def has_stage_state_cost(self) -> bool:
        """Check whether the problem has a stage state cost."""
        cost_is_set = (
            self.stage_state_cost_weight is not None
            and self.stage_state_cost_weight > 1e-10
        )
        if cost_is_set and self.target_states is None:
            raise ProblemDefinitionError(
                "MPC problem has a stage state cost "
                "but the reference trajectory is undefined"
            )
        return cost_is_set

    def get_transition_state_matrix(self, k) -> np.ndarray:
        """Get state-transition matrix at a given timestep.

        Args:
            k: Index of the timestep.

        Returns:
            State-transition matrix at that step.
        """
        return (
            self.transition_state_matrix[k]
            if isinstance(self.transition_state_matrix, list)
            else self.transition_state_matrix
        )

    def get_transition_input_matrix(self, k) -> np.ndarray:
        """Get input-transition matrix at a given timestep.

        Args:
            k: Index of the timestep.

        Returns:
            Input-transition matrix at that step.
        """
        return (
            self.transition_input_matrix[k]
            if isinstance(self.transition_input_matrix, list)
            else self.transition_input_matrix
        )

    def get_ineq_state_matrix(self, k) -> np.ndarray:
        """Get state inequality matrix at a given timestep.

        Args:
            k: Index of the timestep.

        Returns:
            State inequality matrix at that step.
        """
        return (
            self.ineq_state_matrix[k]
            if isinstance(self.ineq_state_matrix, list)
            else self.ineq_state_matrix
        )

    def get_ineq_input_matrix(self, k) -> np.ndarray:
        """Get input inequality matrix at a given timestep.

        Args:
            k: Index of the timestep.

        Returns:
            Input inequality matrix at that step.
        """
        return (
            self.ineq_input_matrix[k]
            if isinstance(self.ineq_input_matrix, list)
            else self.ineq_input_matrix
        )

    def get_ineq_vector(self, k) -> np.ndarray:
        """Get inequality vector at a given timestep.

        Args:
            k: Index of the timestep.

        Returns:
            Inequality vector at that step.
        """
        return (
            self.ineq_vector[k]
            if isinstance(self.ineq_vector, list)
            else self.ineq_vector
        )

    def update_goal_state(self, goal_state: np.ndarray) -> None:
        """Set the goal state.

        Args:
            goal_state: New goal state.

        Raises:
            StateError: if the goal state does not have the right dimension.
        """
        if goal_state.size != self.state_dim:
            raise StateError(
                f"goal state of shape {goal_state.shape} "
                f"does not match state dimension ({self.state_dim})"
            )
        self.goal_state = goal_state.flatten()

    def update_initial_state(self, initial_state: np.ndarray) -> None:
        """Set the initial state.

        Args:
            initial_state: New initial state.

        Raises:
            StateError: if the initial state does not have the right dimension.
        """
        if initial_state.size != self.state_dim:
            raise StateError(
                f"Initial state of shape {initial_state.shape} "
                f"does not match state dimension ({self.state_dim})"
            )
        self.initial_state = initial_state.flatten()

    def update_target_states(self, target_states: np.ndarray) -> None:
        """Set the reference state trajectory to track.

        Args:
            target_states: Reference state trajectory.

        Raises:
            StateError: if the initial state does not have the right dimension.
        """
        if target_states.size != self.state_dim * self.nb_timesteps:
            raise StateError(
                f"Reference state trajectory of shape {target_states.shape} "
                "does not match nb_timesteps * state dimension = "
                f"{self.nb_timesteps} * {self.state_dim} = "
                f"{self.nb_timesteps * self.state_dim}"
            )
        self.target_states = target_states.flatten()

    def __repr__(self) -> str:
        """String representation of the MPC problem."""
        return (
            f"MPCProblem("
            f"goal_state={self.goal_state}, "
            f"ineq_input_matrix={self.ineq_input_matrix}, "
            f"ineq_state_matrix={self.ineq_state_matrix}, "
            f"ineq_vector={self.ineq_vector}, "
            f"initial_state={self.initial_state}, "
            f"input_dim={self.input_dim}, "
            f"nb_timesteps={self.nb_timesteps}, "
            f"stage_input_cost_weight={self.stage_input_cost_weight}, "
            f"stage_state_cost_weight={self.stage_state_cost_weight}, "
            f"state_dim={self.state_dim}, "
            f"terminal_cost_weight={self.terminal_cost_weight}, "
            f"transition_input_matrix={self.transition_input_matrix}, "
            f"transition_state_matrix={self.transition_state_matrix})"
        )
