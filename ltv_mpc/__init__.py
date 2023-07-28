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

"""Linear time-variant model predictive control in Python."""

from .mpc_problem import MPCProblem
from .mpc_qp import MPCQP
from .plan import Plan
from .solve_mpc import solve_mpc

__all__ = [
    "MPCProblem",
    "MPCQP",
    "Plan",
    "solve_mpc",
]

__version__ = "2.0.0"
