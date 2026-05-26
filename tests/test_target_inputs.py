#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the per-step input-reference tracking."""

import unittest

import numpy as np

from qpmpc import MPCProblem, solve_mpc
from qpmpc.exceptions import StateError
from qpmpc.mpc_qp import MPCQP


def _build_double_integrator(
    nb_timesteps: int = 20,
    dt: float = 0.1,
    target_states: np.ndarray | None = None,
    target_inputs: np.ndarray | None = None,
) -> MPCProblem:
    """A scalar second-order system: pos/vel state, jerk input."""
    state_matrix = np.array([[1.0, dt], [0.0, 1.0]])
    input_matrix = np.array([[dt**2 / 2.0], [dt]])
    return MPCProblem(
        transition_state_matrix=state_matrix,
        transition_input_matrix=input_matrix,
        ineq_state_matrix=None,
        ineq_input_matrix=None,
        ineq_vector=[np.array([1.0]) * 1e6] * nb_timesteps,
        nb_timesteps=nb_timesteps,
        terminal_cost_weight=1.0,
        stage_state_cost_weight=1.0,
        stage_input_cost_weight=1e-3,
        initial_state=np.zeros(2),
        goal_state=np.array([1.0, 0.0]),
        target_states=target_states,
        target_inputs=target_inputs,
    )


class TestTargetInputs(unittest.TestCase):
    """Behavioural and bookkeeping tests for input-reference tracking."""

    # ------------------------------------------------------------------ #
    # Field defaults / accessors                                         #
    # ------------------------------------------------------------------ #

    def test_target_inputs_defaults_to_none(self):
        problem = _build_double_integrator()
        self.assertIsNone(problem.target_inputs)
        self.assertFalse(problem.has_stage_input_target)

    def test_constructor_kwarg_persists(self):
        target = np.linspace(0.1, 0.5, 20).reshape(-1, 1)
        problem = _build_double_integrator(target_inputs=target)
        self.assertIsNotNone(problem.target_inputs)
        self.assertTrue(problem.has_stage_input_target)
        np.testing.assert_allclose(problem.target_inputs, target.flatten())

    def test_update_target_inputs_setter(self):
        problem = _build_double_integrator()
        target = np.full((20, 1), 0.25)
        problem.update_target_inputs(target)
        self.assertTrue(problem.has_stage_input_target)
        np.testing.assert_allclose(problem.target_inputs, target.flatten())

    def test_update_target_inputs_shape_check(self):
        problem = _build_double_integrator()
        with self.assertRaises(StateError):
            problem.update_target_inputs(np.zeros(5))  # wrong size

    # ------------------------------------------------------------------ #
    # Cost-vector wiring                                                 #
    # ------------------------------------------------------------------ #

    def test_q_increment_equals_minus_w_u_ref(self):
        """Setting ``target_inputs`` adds exactly ``-w_u · U_ref`` to ``q``."""
        nb = 20
        zero_states = np.zeros(nb * 2)
        without = _build_double_integrator(
            nb_timesteps=nb, target_states=zero_states
        )
        target = np.linspace(0.05, 0.35, nb).reshape(-1, 1)
        with_ref = _build_double_integrator(
            nb_timesteps=nb,
            target_states=zero_states,
            target_inputs=target,
        )
        q_without = MPCQP(without).q.copy()
        q_with = MPCQP(with_ref).q.copy()
        np.testing.assert_allclose(
            q_with - q_without,
            -with_ref.stage_input_cost_weight * target.flatten(),
            atol=1e-12,
        )

    def test_p_matrix_unchanged_by_target_inputs(self):
        """Target inputs only modify the linear cost."""
        nb = 20
        zero_states = np.zeros(nb * 2)
        without = _build_double_integrator(
            nb_timesteps=nb, target_states=zero_states
        )
        target = np.full((nb, 1), 0.42)
        with_ref = _build_double_integrator(
            nb_timesteps=nb,
            target_states=zero_states,
            target_inputs=target,
        )
        np.testing.assert_allclose(
            MPCQP(without).P, MPCQP(with_ref).P, atol=1e-12
        )

    # ------------------------------------------------------------------ #
    # Closed-loop behaviour                                              #
    # ------------------------------------------------------------------ #

    def test_constant_input_reference_pulls_inputs_toward_it(self):
        """With a small state cost, a heavy input reference attracts u."""
        nb = 20
        target_state = np.tile(np.array([0.0, 0.0]), nb).reshape(-1)
        target_input = np.full((nb, 1), 0.5)

        # Heavy input cost dominates over the (zero-target) state cost.
        problem = MPCProblem(
            transition_state_matrix=np.array([[1.0, 0.1], [0.0, 1.0]]),
            transition_input_matrix=np.array([[0.005], [0.1]]),
            ineq_state_matrix=None,
            ineq_input_matrix=None,
            ineq_vector=[np.array([1.0]) * 1e6] * nb,
            nb_timesteps=nb,
            terminal_cost_weight=1e-6,
            stage_state_cost_weight=1e-6,
            stage_input_cost_weight=1.0,
            initial_state=np.zeros(2),
            goal_state=np.zeros(2),
            target_states=target_state,
            target_inputs=target_input,
        )
        plan = solve_mpc(problem, solver="proxqp")
        np.testing.assert_allclose(
            plan.inputs.flatten(), target_input.flatten(), atol=1e-3
        )

    def test_zero_target_inputs_matches_no_target(self):
        """A zero ``target_inputs`` recovers the plain ``||U||²`` objective."""
        nb = 20
        no_target = _build_double_integrator(
            nb_timesteps=nb, target_states=np.zeros(nb * 2)
        )
        zero_target = _build_double_integrator(
            nb_timesteps=nb,
            target_states=np.zeros(nb * 2),
            target_inputs=np.zeros((nb, 1)),
        )
        plan_a = solve_mpc(no_target, solver="proxqp")
        plan_b = solve_mpc(zero_target, solver="proxqp")
        np.testing.assert_allclose(plan_a.inputs, plan_b.inputs, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
