#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

"""Exceptions raised by this library."""


class QPMPCException(Exception):
    """Base class for exceptions from this library."""


class ProblemDefinitionError(QPMPCException):
    """Problem definition is incorrect."""


class PlanError(QPMPCException):
    """Plan is not correct."""


class StateError(QPMPCException):
    """Report an ill-formed state."""
