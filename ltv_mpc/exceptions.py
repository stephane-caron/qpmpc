#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria
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

"""Exceptions raised by this library."""


class LTVMPCException(Exception):
    """Base class for exceptions from ltv-mpc."""


class ProblemDefinitionError(LTVMPCException):
    """Problem definition is incorrect."""


class PlanError(LTVMPCException):
    """Plan is not correct."""


class StateError(LTVMPCException):
    """Report an ill-formed state."""
