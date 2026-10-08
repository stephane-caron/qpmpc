#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""Regression for updating constraints with no state inequalities."""

import unittest

import numpy as np

from qpmpc import MPCProblem
from qpmpc.mpc_qp import MPCQP


class TestInputOnlyConstraintUpdate(unittest.TestCase):
    """Keep actuator bounds independent of the measured state."""

    def test_input_only_constraint_update(self):
        """Updating the state must preserve input-only constraint bounds."""
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                problem = MPCProblem(
                    transition_state_matrix=np.eye(1),
                    transition_input_matrix=np.eye(1),
                    ineq_state_matrix=None,
                    ineq_input_matrix=np.array([[1.0], [-1.0]]),
                    ineq_vector=np.array([0.5, 0.5]),
                    nb_timesteps=2,
                    terminal_cost_weight=1.0,
                    stage_state_cost_weight=None,
                    stage_input_cost_weight=1.0,
                    initial_state=np.array([0.0]),
                    goal_state=np.array([1.0]),
                )
                qp = MPCQP(problem, sparse=sparse)
                initial_bounds = qp.h.copy()
                problem.update_initial_state(np.array([0.25]))
                qp.update_constraint_vector(problem)
                self.assertIsNone(qp.C)
                np.testing.assert_array_equal(qp.h, initial_bounds)
                np.testing.assert_array_equal(qp.h, [0.5] * 4)


if __name__ == "__main__":
    unittest.main()
