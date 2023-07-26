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

import unittest
from dataclasses import dataclass

import numpy as np

from ltv_mpc import MPCProblem, solve_mpc
from ltv_mpc.solve_mpc import MPCQP

gravity = 9.81  # [m] / [s]^2


@dataclass
class HumanoidSteppingProblem:

    com_height: float = 0.8
    dsp_duration: float = 0.1
    end_pos: float = 0.3
    foot_length: float = 0.1
    horizon_duration: float = 2.5
    nb_timesteps: int = 16
    ssp_duration: float = 0.7
    start_pos: float = 0.0


class TestHumanoid(unittest.TestCase):
    def setUp(self):
        """
        Prepare test fixture.
        """
        problem = HumanoidSteppingProblem()
        T = problem.horizon_duration / problem.nb_timesteps
        nb_init_dsp_steps = int(round(problem.dsp_duration / T))
        nb_init_ssp_steps = int(round(problem.ssp_duration / T))
        nb_dsp_steps = int(round(problem.dsp_duration / T))
        state_matrix = np.array(
            [[1.0, T, T ** 2 / 2.0], [0.0, 1.0, T], [0.0, 0.0, 1.0]]
        )
        input_matrix = np.array([T ** 3 / 6.0, T ** 2 / 2.0, T])
        input_matrix = input_matrix.reshape((3, 1))
        zmp_from_state = np.array([1.0, 0.0, -problem.com_height / gravity])
        ineq_matrix = np.array([+zmp_from_state, -zmp_from_state])
        cur_max = problem.start_pos + 0.5 * problem.foot_length
        cur_min = problem.start_pos - 0.5 * problem.foot_length
        next_max = problem.end_pos + 0.5 * problem.foot_length
        next_min = problem.end_pos - 0.5 * problem.foot_length
        ineq_vector = [
            np.array([+1000.0, +1000.0])
            if i < nb_init_dsp_steps
            else np.array([+cur_max, -cur_min])
            if i - nb_init_dsp_steps <= nb_init_ssp_steps
            else np.array([+1000.0, +1000.0])
            if i - nb_init_dsp_steps - nb_init_ssp_steps < nb_dsp_steps
            else np.array([+next_max, -next_min])
            for i in range(problem.nb_timesteps)
        ]
        mpc_problem = MPCProblem(
            transition_state_matrix=state_matrix,
            transition_input_matrix=input_matrix,
            ineq_state_matrix=ineq_matrix,
            ineq_input_matrix=None,
            ineq_vector=ineq_vector,
            initial_state=np.array([problem.start_pos, 0.0, 0.0]),
            goal_state=np.array([problem.end_pos, 0.0, 0.0]),
            nb_timesteps=problem.nb_timesteps,
            terminal_cost_weight=1.0,
            stage_state_cost_weight=None,
            stage_input_cost_weight=1e-3,
        )
        #
        self.stepping_problem = problem
        self.problem = mpc_problem

    def test_mpc_qp_first_constraints(self):
        mpc_qp = MPCQP(self.problem)
        qp = mpc_qp.problem
        self.assertAlmostEqual(np.linalg.norm(qp.G[0:2]), 0.0)

    def test_solve_mpc(self):
        solution = solve_mpc(self.problem, solver="quadprog")
        N = self.problem.nb_timesteps
        input_dim = self.problem.input_dim
        state_dim = self.problem.state_dim
        U = solution.inputs
        X = solution.states
        self.assertEqual(U.flatten().shape, (N * input_dim,))
        self.assertEqual(X.flatten().shape, ((N + 1) * state_dim,))


if __name__ == "__main__":
    unittest.main()
