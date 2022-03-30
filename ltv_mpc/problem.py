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
        if type(transition_input_matrix) is np.ndarray:
            self.input_dim = transition_input_matrix.shape[1]
        else:  # type(transition_input_matrix) is List[np.ndarray]
            self.input_dim = transition_input_matrix[0].shape[1]
        if type(transition_state_matrix) is np.ndarray:
            self.state_dim = transition_state_matrix.shape[1]
        else:  # type(transition_state_matrix) is List[np.ndarray]
            self.state_dim = transition_state_matrix[0].shape[1]

    def get_transition_state_matrix(self, k):
        return (
            self.transition_state_matrix[k]
            if type(self.transition_state_matrix) is list
            else self.transition_state_matrix
        )

    def get_transition_input_matrix(self, k):
        return (
            self.transition_input_matrix[k]
            if type(self.transition_input_matrix) is list
            else self.transition_input_matrix
        )

    def get_ineq_state_matrix(self, k):
        return (
            self.ineq_state_matrix[k]
            if type(self.ineq_state_matrix) is list
            else self.ineq_state_matrix
        )

    def get_ineq_input_matrix(self, k):
        return (
            self.ineq_input_matrix[k]
            if type(self.ineq_input_matrix) is list
            else self.ineq_input_matrix
        )

    def get_ineq_vector(self, k):
        return (
            self.ineq_vector[k]
            if type(self.ineq_vector) is list
            else self.ineq_vector
        )
