# SPDX-License-Identifier: Apache-2.0

"""Regression for the exact pendulum input at short sampling periods."""

import unittest
from decimal import Decimal, localcontext

import numpy as np

from qpmpc.systems import WheeledInvertedPendulum


def input_angle_reference(
    period: float, length: float, gravity: float
) -> float:
    """Integrate theta'' = (g/l) theta - 1/l from a zero state.

    The power series follows the differential equation rather than the
    hyperbolic expression used by the discretization implementation.
    """
    with localcontext() as context:
        context.prec = 80
        dt = Decimal.from_float(period)
        ell = Decimal.from_float(length)
        acceleration = Decimal.from_float(gravity)
        term = -(dt * dt) / (2 * ell)
        result = term
        for order in range(1, 100):
            term *= (
                (acceleration / ell)
                * dt
                * dt
                / ((2 * order + 1) * (2 * order + 2))
            )
            result += term
        return float(result)


class TestPendulumInputDiscretization(unittest.TestCase):
    """Preserve the exact zero-order-hold input coupling at small steps."""

    def test_input_angle_at_short_periods(self) -> None:
        """The input must agree with the ODE and half-step composition."""
        for length in (0.6, 2.0):
            for period in (0.0, 1e-12, 1e-9, 1e-7, 1e-4, 0.1, 1.0):
                with self.subTest(length=length, period=period):
                    pendulum = WheeledInvertedPendulum(
                        length=length, sampling_period=period
                    )
                    problem = pendulum.build_mpc_problem()
                    expected = input_angle_reference(
                        period, length, pendulum.GRAVITY
                    )
                    np.testing.assert_allclose(
                        problem.transition_input_matrix[1, 0],
                        expected,
                        rtol=2e-14,
                        atol=0.0,
                    )
                    half_step = WheeledInvertedPendulum(
                        length=length, sampling_period=period / 2
                    ).build_mpc_problem()
                    np.testing.assert_allclose(
                        problem.transition_input_matrix,
                        half_step.transition_state_matrix
                        @ half_step.transition_input_matrix
                        + half_step.transition_input_matrix,
                        rtol=2e-14,
                        atol=0.0,
                    )


if __name__ == "__main__":
    unittest.main()
