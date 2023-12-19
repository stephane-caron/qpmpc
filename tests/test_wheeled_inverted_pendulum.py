#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Stéphane Caron
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from qpmpc import solve_mpc
from qpmpc.systems import WheeledInvertedPendulum


class TestWheeledInvertedPendulum(unittest.TestCase):
    def setUp(self):
        self.pendulum = WheeledInvertedPendulum()

    def test_properties(self):
        self.assertGreater(self.pendulum.horizon_duration, 0.1)
        self.assertGreater(self.pendulum.omega, 0.1)

    def test_mpc_problem(self):
        mpc_problem = self.pendulum.build_mpc_problem(
            terminal_cost_weight=10.0,
            stage_state_cost_weight=1.0,
            stage_input_cost_weight=1e-3,
        )
        initial_state = np.zeros(self.pendulum.STATE_DIM)
        goal_state = initial_state.copy()
        nx = self.pendulum.STATE_DIM
        target_states = np.zeros(self.pendulum.nb_timesteps * nx)
        mpc_problem.update_initial_state(initial_state)
        mpc_problem.update_goal_state(goal_state)
        mpc_problem.update_target_states(target_states)
        plan = solve_mpc(mpc_problem, solver="proxqp")
        self.assertIsNotNone(plan)
        self.assertIsNotNone(plan.first_input)
        dt = self.pendulum.sampling_period
        state = self.pendulum.integrate(initial_state, plan.first_input, dt)
        self.assertTrue(np.allclose(state, initial_state))


if __name__ == "__main__":
    unittest.main()
