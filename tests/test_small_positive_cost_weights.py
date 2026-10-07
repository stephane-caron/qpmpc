#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""Physical MPC objectives retain their terms at every positive scale."""

import unittest

import numpy as np
from qpsolvers import Problem, solve_problem

from qpmpc import MPCProblem
from qpmpc.exceptions import ProblemDefinitionError
from qpmpc.mpc_qp import MPCQP

SCALES = (1e-16, 1e-12, 1e-10, 1e-8, 1., 1e8)
MODES = ("terminal", "stage", "both")


def make_problem(scale=1., mode="both"):
    """Build a coupled LTV tracking problem with one common cost scale."""
    terminal_weight = None if mode == "stage" else 4. * scale
    stage_weight = None if mode == "terminal" else .7 * scale
    return MPCProblem(
        transition_state_matrix=[
            np.array([[1., .2], [.1, .9]]),
            np.array([[.9, .15], [0., 1.05]]),
            np.array([[1.1, .05], [.1, .95]]),
        ],
        transition_input_matrix=[
            np.array([[.4, .1], [.05, .3]]),
            np.array([[.3, -.05], [.1, .35]]),
            np.array([[.35, .05], [-.1, .4]]),
        ],
        ineq_state_matrix=np.zeros((4, 2)),
        ineq_input_matrix=np.vstack([np.eye(2), -np.eye(2)]),
        ineq_vector=np.full(4, 5.),
        nb_timesteps=3,
        terminal_cost_weight=terminal_weight,
        stage_state_cost_weight=stage_weight,
        stage_input_cost_weight=.3 * scale,
        initial_state=np.array([.2, -.15]),
        goal_state=np.array([.8, -.35]),
        target_states=np.array([[.3, -.1], [.4, -.15], [.5, -.2]]),
        target_inputs=np.array([[.05, -.02], [.1, .03], [-.04, .02]]),
    )


def physical_objective(problem, inputs):
    """Evaluate the original scalar objective without the cost flags."""
    controls = inputs.reshape(3, 2)
    state = problem.initial_state.copy()
    state_cost = 0.
    reference_states = problem.target_states.reshape(3, 2)
    for k, control in enumerate(controls):
        error = state - reference_states[k]
        state_cost += error @ error
        state = (problem.transition_state_matrix[k] @ state
                 + problem.transition_input_matrix[k] @ control)
    terminal_error = state - problem.goal_state
    input_error = controls - problem.target_inputs.reshape(3, 2)
    terminal_weight = problem.terminal_cost_weight or 0.
    state_weight = problem.stage_state_cost_weight or 0.
    return .5 * (terminal_weight * (terminal_error @ terminal_error)
                 + state_weight * state_cost
                 + problem.stage_input_cost_weight
                 * np.sum(input_error * input_error))


def physical_coefficients(problem):
    """Recover gradient and Hessian from independent scalar evaluations."""
    basis = np.eye(6)
    constant = physical_objective(problem, np.zeros(6))
    plus = np.array([physical_objective(problem, row) for row in basis])
    minus = np.array([physical_objective(problem, -row) for row in basis])
    gradient = (plus - minus) / 2.
    hessian = np.diag(plus + minus - 2. * constant)
    for i in range(6):
        for j in range(i):
            mixed = (physical_objective(problem, basis[i] + basis[j])
                     - plus[i] - plus[j] + constant)
            hessian[i, j] = hessian[j, i] = mixed
    return hessian, gradient


class TestSmallPositiveCostWeights(unittest.TestCase):
    """Keep the condensed objective consistent with the physical objective."""

    def test_raw_qp_coefficients_match_physical_gradient_and_hessian(self):
        """Include tiny positive terminal and stage costs in both P and q."""
        for mode in MODES:
            for scale in SCALES:
                for sparse in (False, True):
                    with self.subTest(mode=mode, scale=scale, sparse=sparse):
                        problem = make_problem(scale, mode)
                        qp = MPCQP(problem, sparse=sparse)
                        expected_p, expected_q = physical_coefficients(problem)
                        actual_p = qp.P.toarray() if sparse else qp.P
                        np.testing.assert_allclose(
                            actual_p, expected_p, rtol=2e-11,
                            atol=scale * 2e-11)
                        np.testing.assert_allclose(
                            qp.q, expected_q, rtol=2e-11,
                            atol=scale * 2e-11)
                        point = np.linspace(-.2, .3, 6)
                        np.testing.assert_allclose(
                            actual_p @ point + qp.q,
                            expected_p @ point + expected_q,
                            rtol=2e-11, atol=scale * 2e-11)

    def test_positive_global_cost_rescaling_preserves_analytic_optimum(self):
        """Positive rescaling cannot change a strictly convex minimizer."""
        for mode in MODES:
            expected_p, expected_q = physical_coefficients(make_problem(1., mode))
            expected = np.linalg.solve(expected_p, -expected_q)
            for scale in SCALES:
                with self.subTest(mode=mode, scale=scale):
                    qp = MPCQP(make_problem(scale, mode))
                    actual = np.linalg.solve(qp.P, -qp.q)
                    np.testing.assert_allclose(actual, expected,
                                               rtol=2e-11, atol=2e-11)

    def test_normalized_proxqp_obeys_physical_optimum_and_kkt(self):
        """Solve explicit equivalent normalized QPs, not tiny-scale QPs."""
        # Division by a positive scale leaves the minimizer unchanged. It
        # avoids claiming an absolute precision for unnormalized tiny costs.
        for mode in MODES:
            expected_p, expected_q = physical_coefficients(make_problem(1., mode))
            expected = np.linalg.solve(expected_p, -expected_q)
            for scale in SCALES:
                with self.subTest(mode=mode, scale=scale):
                    qp = MPCQP(make_problem(scale, mode))
                    normalized = Problem(qp.P / scale, qp.q / scale,
                                         qp.G, qp.h)
                    solution = solve_problem(normalized, solver="proxqp",
                                             eps_abs=1e-10, eps_rel=1e-10)
                    self.assertTrue(solution.found)
                    np.testing.assert_allclose(solution.x, expected,
                                               rtol=2e-8, atol=2e-8)
                    self.assertTrue(np.all(qp.G @ solution.x < qp.h - 1.))
                    np.testing.assert_allclose(
                        expected_p @ solution.x + expected_q,
                        0., rtol=0., atol=2e-8)

    def test_tiny_cost_updates_match_new_measurement_and_references(self):
        """Check real updates against freshly evaluated physical objectives."""
        for mode in MODES:
            for scale in (1e-16, 1e-12, 1e-10):
                with self.subTest(mode=mode, scale=scale):
                    problem = make_problem(scale, mode)
                    qp = MPCQP(problem)
                    problem.update_initial_state(np.array([-.1, .2]))
                    problem.update_goal_state(np.array([.6, .1]))
                    problem.update_target_states(
                        np.array([[.1, .2], [.3, .1], [.5, 0.]]))
                    qp.update_cost_vector(problem)
                    expected_p, expected_q = physical_coefficients(problem)
                    np.testing.assert_allclose(qp.P, expected_p,
                                               rtol=2e-11,
                                               atol=scale * 2e-11)
                    np.testing.assert_allclose(qp.q, expected_q,
                                               rtol=2e-11,
                                               atol=scale * 2e-11)

    def test_tiny_terminal_weight_requires_a_goal(self):
        """Every positive terminal term requires its physical reference."""
        for scale in (1e-16, 1e-12):
            with self.subTest(scale=scale):
                problem = make_problem(scale, "terminal")
                problem.goal_state = None
                with self.assertRaises(ProblemDefinitionError):
                    MPCQP(problem)

    def test_tiny_stage_weight_requires_a_reference_trajectory(self):
        """Every positive stage term requires its physical reference."""
        for scale in (1e-16, 1e-12):
            with self.subTest(scale=scale):
                problem = make_problem(scale, "stage")
                problem.target_states = None
                with self.assertRaises(ProblemDefinitionError):
                    MPCQP(problem)

    def test_none_and_zero_weights_remain_disabled_without_references(self):
        """Preserve the explicit disabled-cost contract and input objective."""
        for terminal, stage in ((0., None), (None, 0.), (0., 0.)):
            with self.subTest(terminal=terminal, stage=stage):
                problem = make_problem()
                problem.terminal_cost_weight = terminal
                problem.stage_state_cost_weight = stage
                problem.goal_state = None
                problem.target_states = None
                self.assertFalse(problem.has_terminal_cost)
                self.assertFalse(problem.has_stage_state_cost)
                qp = MPCQP(problem)
                np.testing.assert_array_equal(qp.P, .3 * np.eye(6))
                np.testing.assert_array_equal(qp.q, -.3 * problem.target_inputs)


if __name__ == "__main__":
    unittest.main()
