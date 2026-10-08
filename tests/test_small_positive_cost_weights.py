# SPDX-License-Identifier: Apache-2.0

"""Regression for consistent terminal and stage state cost activation."""

import unittest

import numpy as np

from qpmpc import MPCProblem
from qpmpc.exceptions import ProblemDefinitionError
from qpmpc.mpc_qp import MPCQP


class TestSmallPositiveCostWeights(unittest.TestCase):
    """Reject negative state weights and retain positive tracking terms."""

    def test_state_cost_weights(self):
        """Use the same cost activation rule for P, q, and references."""
        for cost in ("terminal", "stage"):
            for weight in (-1e-12, 0.0, 1e-12):
                with self.subTest(cost=cost, weight=weight):
                    kwargs = dict(
                        transition_state_matrix=np.eye(1),
                        transition_input_matrix=np.eye(1),
                        ineq_state_matrix=None,
                        ineq_input_matrix=None,
                        ineq_vector=np.empty(0),
                        nb_timesteps=2,
                        terminal_cost_weight=(
                            weight if cost == "terminal" else None
                        ),
                        stage_state_cost_weight=(
                            weight if cost == "stage" else None
                        ),
                        stage_input_cost_weight=1e-12,
                        initial_state=np.array([0.0]),
                        goal_state=np.array([1.0]) if weight > 0.0 else None,
                        target_states=(
                            np.array([0.0, 2.0]) if weight > 0.0 else None
                        ),
                    )
                    if weight < 0.0:
                        with self.assertRaises(ProblemDefinitionError):
                            MPCProblem(**kwargs)
                        continue
                    problem = MPCProblem(**kwargs)
                    qp = MPCQP(problem)
                    expected_p = np.eye(2)
                    expected_q = np.zeros(2)
                    if weight > 0.0:
                        if cost == "terminal":
                            expected_p += np.ones((2, 2))
                            expected_q = np.array([-1.0, -1.0])
                        else:
                            expected_p[0, 0] += 1.0
                            expected_q = np.array([-2.0, 0.0])
                    np.testing.assert_allclose(
                        qp.P / 1e-12, expected_p, rtol=0.0, atol=1e-14
                    )
                    np.testing.assert_allclose(
                        qp.q / 1e-12, expected_q, rtol=0.0, atol=1e-14
                    )
                    if weight > 0.0:
                        if cost == "terminal":
                            problem.goal_state = None
                        else:
                            problem.target_states = None
                        with self.assertRaises(ProblemDefinitionError):
                            MPCQP(problem)


if __name__ == "__main__":
    unittest.main()
