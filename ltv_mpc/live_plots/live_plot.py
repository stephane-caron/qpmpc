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

"""Utility functions and classes."""

from typing import Any, Dict, Sequence

import matplotlib
from matplotlib import pyplot as plt

from ..exceptions import LTVMPCException


class LivePlot:
    """Live plot using matplotlib."""

    lines: Dict[str, Any]

    def __init__(self, xlim, ylim, ylim2=None, fast: bool = True):
        """Initialize live plot.

        Args:
            xlim: Limits for the x-axis.
            ylim: Limits for left y-axis.
            ylim2: Limits for the right y-axis.
            fast: If set, use blitting.
        """
        if fast:  # blitting doesn't work with all matplotlib backends
            matplotlib.use("TkAgg")
        figure, axis = plt.subplots()
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        rhs_axis = None
        if ylim2 is not None:
            rhs_axis = axis.twinx()
            rhs_axis.set_ylim(*ylim2)
        plt.show(block=False)
        plt.pause(0.05)
        self.axis = axis
        self.background = None
        self.canvas = figure.canvas
        self.canvas.mpl_connect("draw_event", self.__on_draw)
        self.fast = fast
        self.figure = figure
        self.lines = {}
        self.rhs_axis = rhs_axis

    def add_line(self, name, *args, **kwargs) -> None:
        """Add a line-plot to the left axis.

        Args:
            name: Name to refer to this line, for updates.
            args: Forwarded to ``pyplot.plot``.
            kwargs: Forwarded to ``pyplot.plot``.
        """
        kwargs["animated"] = True
        (line,) = self.axis.plot([], *args, **kwargs)
        self.lines[name] = line

    def add_rhs_line(self, name, *args, **kwargs) -> None:
        """Add a line-plot to the right axis.

        Args:
            name: Name to refer to this line, for updates.
            args: Forwarded to ``pyplot.plot``.
            kwargs: Forwarded to ``pyplot.plot``.
        """
        if self.rhs_axis is None:
            raise LTVMPCException("right-hand side axis not initialized")
        kwargs["animated"] = True
        (line,) = self.rhs_axis.plot([], *args, **kwargs)
        self.lines[name] = line

    def legend(self, legend: Sequence[str]) -> None:
        """Add a legend to the plot.

        Args:
            legend: Legend.
        """
        self.axis.legend(legend)

    def update_line(self, name: str, xdata, ydata) -> None:
        """Update a previously-added line.

        Args:
            name: Name of the line to update.
            xdata: New x-axis data.
            ydata: New y-axis data.
        """
        self.lines[name].set_data(xdata, ydata)

    def __draw_lines(self):
        for line in self.lines.values():
            self.figure.draw_artist(line)

    def __on_draw(self, event):
        if event is not None:
            if event.canvas != self.canvas:
                raise RuntimeError
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)
        self.__draw_lines()

    def update(self) -> None:
        """Update the output figure."""
        if self.background is None:
            self.__on_draw(None)
        elif self.fast:
            self.canvas.restore_region(self.background)
            self.__draw_lines()
            self.canvas.blit(self.figure.bbox)
        else:  # slow mode, if blitting doesn't work
            self.canvas.draw()
        self.canvas.flush_events()
