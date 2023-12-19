#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Stéphane Caron
# SPDX-License-Identifier: Apache-2.0

"""Solve model predictive control problems."""

from qpsolvers import solve_problem

from .mpc_problem import MPCProblem
from .mpc_qp import MPCQP
from .plan import Plan


def solve_mpc(
    problem: MPCProblem,
    solver: str,
    sparse: bool = False,
    **kwargs,
) -> Plan:
    """Solve a linear time-invariant model predictive control problem.

    Args:
        problem: Model predictive control problem to solve.
        solver: Quadratic programming solver to use, to choose in
            :data:`qpsolvers.available_solvers`. Empirically the best
            performing solvers are Clarabel and ProxQP: see for instance this
            `benchmark of QP solvers for model predictive control
            <https://github.com/qpsolvers/mpc_qpbenchmark>`__.
        sparse: Whether to use sparse or dense matrices in the output quadratic
            program. Enable it if the QP solver is sparse (e.g. OSQP).
        kwargs: Keyword arguments forwarded to the QP solver via the
            `solve_qp`_ function.

    Returns:
        Solution to the problem, if found.

    .. _solve_qp:
        https://qpsolvers.github.io/qpsolvers/quadratic-programming.html#qpsolvers.solve_qp
    """
    mpc_qp = MPCQP(problem, sparse=sparse)
    qpsol = solve_problem(mpc_qp.problem, solver=solver, **kwargs)
    return Plan(problem, qpsol)
