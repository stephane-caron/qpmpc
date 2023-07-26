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
            :data:`qpsolvers.available_solvers`. Both "quadprog" and "osqp"
            tend to perform well on model predictive control problems. See for
            instance `this benchmark
            <https://github.com/qpsolvers/qpsolvers#benchmark>`__.
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
