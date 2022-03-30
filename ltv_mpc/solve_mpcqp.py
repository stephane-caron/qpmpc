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

from qpsolvers import solve_qp

from .mpcqp import MPCQP
from .problem import Problem
from .solution import Solution


def solve_mpcqp(problem: Problem, qp: MPCQP, **kwargs) -> Solution:
    U = solve_qp(
        qp.cost_matrix,
        qp.cost_vector,
        qp.ineq_matrix,
        qp.ineq_vector,
        **kwargs
    )
    U = U.reshape((problem.nb_timesteps, problem.input_dim))
    return Solution(problem, U)
