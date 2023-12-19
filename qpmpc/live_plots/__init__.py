#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria
# SPDX-License-Identifier: Apache-2.0

"""Set of system-specific live plots provided for reference and examples."""

from .live_plot import LivePlot
from .wheeled_inverted_pendulum_plot import WheeledInvertedPendulumPlot

__all__ = [
    "WheeledInvertedPendulumPlot",
    "LivePlot",
]
