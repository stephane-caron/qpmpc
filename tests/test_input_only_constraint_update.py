#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""Receding-horizon constraints checked against uncondensed dynamics."""

import itertools
import unittest

import numpy as np
from qpsolvers import solve_problem

from qpmpc import MPCProblem
from qpmpc.exceptions import ProblemDefinitionError
from qpmpc.mpc_qp import MPCQP


def make_problem(kind="input"):
    """Construct a coupled three-state, two-actuator LTV system."""
    transitions = [
        np.array([[1., .2, 0.], [0., .9, .1], [.05, 0., 1.]]),
        np.array([[.95, .15, .05], [.1, 1., 0.], [0., .05, .9]]),
        np.array([[1., .1, 0.], [0., .95, .15], [.1, 0., 1.]]),
    ]
    actuators = [
        np.array([[.3, .05], [.1, .4], [.05, -.1]]),
        np.array([[.2, .1], [.05, .35], [.1, -.05]]),
        np.array([[.25, .05], [.1, .3], [.05, -.15]]),
    ]
    limits = [np.array([.2, .35]), np.array([.25, .3]),
              np.array([.3, .4])]
    input_matrix = np.vstack([np.eye(2), -np.eye(2)])
    state_matrices = None
    input_matrices = [input_matrix] * 3
    bounds = [np.tile(limit, 2) for limit in limits]
    if kind == "state":
        state_matrices = [
            np.array([[.3, .1, -.05], [-.3, -.1, .05]]) * factor
            for factor in (1., 1.2, .8)
        ]
        input_matrices = None
        bounds = [np.array([.8, .9])] * 3
    elif kind == "mixed":
        state_matrices = [
            np.array([[.1, 0., .05], [0., -.1, 0.],
                      [-.1, 0., -.05], [0., .1, 0.]]) * factor
            for factor in (1., .8, 1.2)
        ]
    elif kind == "empty":
        input_matrices = None
        bounds = [np.empty(0)] * 3
    elif kind != "input":
        raise ValueError(kind)
    return MPCProblem(
        transition_state_matrix=transitions,
        transition_input_matrix=actuators,
        ineq_state_matrix=state_matrices,
        ineq_input_matrix=input_matrices,
        ineq_vector=bounds,
        nb_timesteps=3,
        terminal_cost_weight=6.,
        stage_state_cost_weight=.7,
        stage_input_cost_weight=1.5,
        initial_state=np.array([.1, -.25, .2]),
        goal_state=np.array([1.7, -1.2, .6]),
        target_states=np.array([[.3, -.1, .1], [.5, -.2, .2],
                                [.7, -.3, .1]]),
        target_inputs=np.array([[.05, -.02], [.1, .03], [-.04, .02]]),
    )


def rollout(problem, inputs):
    """Integrate the original dynamics without using Phi or Psi."""
    state = problem.initial_state.copy()
    states = [state]
    for k, control in enumerate(inputs.reshape(3, 2)):
        state = (problem.transition_state_matrix[k] @ state
                 + problem.transition_input_matrix[k] @ control)
        states.append(state)
    return np.array(states)


def physical_cost(problem, inputs):
    """Compute the scalar physical objective before condensation."""
    states = rollout(problem, inputs)
    references = problem.target_states.reshape(3, 3)
    target_inputs = problem.target_inputs.reshape(3, 2)
    state_error = states[:-1] - references
    terminal_error = states[-1] - problem.goal_state
    input_error = inputs.reshape(3, 2) - target_inputs
    return .5 * (problem.terminal_cost_weight * terminal_error.ravel().dot(
        terminal_error.ravel())
        + problem.stage_state_cost_weight * state_error.ravel().dot(
            state_error.ravel())
        + problem.stage_input_cost_weight * input_error.ravel().dot(
            input_error.ravel()))


def independent_quadratic(problem):
    """Recover exact quadratic coefficients from scalar objective values."""
    basis = np.eye(6)
    constant = physical_cost(problem, np.zeros(6))
    positive = np.array([physical_cost(problem, row) for row in basis])
    negative = np.array([physical_cost(problem, -row) for row in basis])
    gradient = (positive - negative) / 2.
    hessian = np.diag(positive + negative - 2. * constant)
    for i in range(6):
        for j in range(i):
            value = (physical_cost(problem, basis[i] + basis[j])
                     - positive[i] - positive[j] + constant)
            hessian[i, j] = hessian[j, i] = value
    return hessian, gradient


def box_optimum(hessian, gradient, limits):
    """Enumerate box active sets using primal feasibility and KKT signs."""
    for pattern in itertools.product((-1, 0, 1), repeat=6):
        pattern = np.array(pattern)
        free = pattern == 0
        active = ~free
        point = pattern * limits
        if np.any(free):
            rhs = -gradient[free] - hessian[np.ix_(free, active)] @ point[active]
            point[free] = np.linalg.solve(hessian[np.ix_(free, free)], rhs)
        derivative = hessian @ point + gradient
        if (np.all(np.abs(point) <= limits + 1e-11)
                and np.all(np.abs(derivative[free]) <= 1e-10)
                and np.all(derivative[pattern == -1] >= -1e-10)
                and np.all(derivative[pattern == 1] <= 1e-10)):
            return point
    raise AssertionError("Strictly convex box QP has no KKT active set")


class TestInputOnlyConstraintUpdate(unittest.TestCase):
    """Preserve null state constraints during receding-horizon updates."""

    def check_residuals(self, kind):
        """Compare condensed residuals with each physical stage inequality."""
        for sparse in (False, True):
            with self.subTest(kind=kind, sparse=sparse):
                problem = make_problem(kind)
                qp = MPCQP(problem, sparse=sparse)
                problem.update_initial_state(np.array([.4, -.15, .35]))
                qp.update_cost_vector(problem)
                qp.update_constraint_vector(problem)
                for inputs in (np.zeros(6), np.linspace(-.15, .2, 6),
                               np.array([.6, -.5, .3, .2, -.4, .7])):
                    states = rollout(problem, inputs)
                    expected = []
                    for k, control in enumerate(inputs.reshape(3, 2)):
                        residual = -problem.ineq_vector[k].copy()
                        if problem.ineq_state_matrix is not None:
                            residual += problem.ineq_state_matrix[k] @ states[k]
                        if problem.ineq_input_matrix is not None:
                            residual += problem.ineq_input_matrix[k] @ control
                        expected.extend(residual)
                    np.testing.assert_allclose(
                        qp.G @ inputs - qp.h, expected, rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(
                        problem.integrate(problem.initial_state,
                                          inputs.reshape(3, 2)),
                        states, rtol=1e-12, atol=1e-12)

    def test_input_only_update_preserves_actuator_residuals(self):
        """Input bounds must not acquire a dependence on the initial state."""
        self.check_residuals("input")

    def test_state_only_update_preserves_physical_constraint_residuals(self):
        """Retain the existing state-dependent constraint update."""
        self.check_residuals("state")

    def test_mixed_state_and_input_constraints_keep_their_stage_alignment(self):
        """Retain aligned LTV state and actuator contributions."""
        self.check_residuals("mixed")

    def test_repeated_input_only_updates_keep_fixed_bounds_and_correct_cost(self):
        """Check physical objective coefficients after each new measurement."""
        for sparse in (False, True):
            problem = make_problem()
            qp = MPCQP(problem, sparse=sparse)
            original_bounds = qp.h.copy()
            for initial in ([.4, -.15, .35], [-.2, .3, -.1], [.1, -.25, .2]):
                with self.subTest(sparse=sparse, initial=initial):
                    problem.update_initial_state(np.array(initial))
                    qp.update_cost_vector(problem)
                    qp.update_constraint_vector(problem)
                    np.testing.assert_array_equal(qp.h, original_bounds)
                    expected_p, expected_q = independent_quadratic(problem)
                    actual_p = qp.P.toarray() if sparse else qp.P
                    np.testing.assert_allclose(actual_p, expected_p,
                                               rtol=2e-12, atol=2e-12)
                    np.testing.assert_allclose(qp.q, expected_q,
                                               rtol=2e-12, atol=2e-12)

    def test_updated_box_qp_matches_independent_optimum_and_kkt_conditions(self):
        """Check real ProxQP against independently enumerated active sets."""
        problem = make_problem()
        qp = MPCQP(problem)
        problem.update_initial_state(np.array([.4, -.15, .35]))
        qp.update_cost_vector(problem)
        qp.update_constraint_vector(problem)
        expected_p, expected_q = independent_quadratic(problem)
        self.assertGreater(np.linalg.eigvalsh(expected_p).min(), 1.)
        limits = np.concatenate([bound[:2] for bound in problem.ineq_vector])
        expected = box_optimum(expected_p, expected_q, limits)
        solution = solve_problem(qp.problem, solver="proxqp",
                                 eps_abs=1e-10, eps_rel=1e-10)
        self.assertTrue(solution.found)
        np.testing.assert_allclose(solution.x, expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_array_less(np.abs(solution.x), limits + 1e-8)
        self.assertTrue(np.any(np.isclose(np.abs(expected), limits, atol=1e-9)))
        gradient = expected_p @ solution.x + expected_q
        lower = np.isclose(solution.x, -limits, atol=1e-8)
        upper = np.isclose(solution.x, limits, atol=1e-8)
        free = ~(lower | upper)
        np.testing.assert_allclose(gradient[free], 0., rtol=0., atol=1e-7)
        self.assertTrue(np.all(gradient[lower] >= -1e-7))
        self.assertTrue(np.all(gradient[upper] <= 1e-7))

    def test_empty_inequality_rows_remain_empty_after_state_update(self):
        """A valid zero-row constraint system must stay unconstrained."""
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                problem = make_problem("empty")
                qp = MPCQP(problem, sparse=sparse)
                problem.update_initial_state(np.array([.4, -.15, .35]))
                qp.update_constraint_vector(problem)
                self.assertEqual(qp.G.shape, (0, 6))
                self.assertEqual(qp.h.shape, (0,))
                self.assertIsNone(qp.C)

    def test_missing_initial_state_still_raises_definition_error(self):
        """Null state constraints do not bypass the initial-state contract."""
        problem = make_problem()
        qp = MPCQP(problem)
        problem.initial_state = None
        with self.assertRaises(ProblemDefinitionError):
            qp.update_constraint_vector(problem)


if __name__ == "__main__":
    unittest.main()
