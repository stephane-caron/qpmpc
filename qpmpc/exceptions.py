#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

"""Exceptions raised by this library."""


class LTVMPCException(Exception):
    """Base class for exceptions from this library."""


class ProblemDefinitionError(LTVMPCException):
    """Problem definition is incorrect."""


class PlanError(LTVMPCException):
    """Plan is not correct."""


class StateError(LTVMPCException):
    """Report an ill-formed state."""
