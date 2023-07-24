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

from typing import Any, Dict

import matplotlib
from matplotlib import pyplot as plt

from .exceptions import LTVMPCException


class LivePlot:

    lines: Dict[str, Any]

    def __init__(self, xlim, ylim, ylim2=None, fast: bool = True):
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
        self.canvas.mpl_connect("draw_event", self.on_draw)
        self.fast = fast
        self.figure = figure
        self.lines = {}
        self.rhs_axis = rhs_axis

    def add_line(self, name, *args, **kwargs):
        kwargs["animated"] = True
        (line,) = self.axis.plot([], *args, **kwargs)
        self.lines[name] = line

    def add_rhs_line(self, name, *args, **kwargs):
        if self.rhs_axis is None:
            raise LTVMPCException("right-hand side axis not initialized")
        kwargs["animated"] = True
        (line,) = self.rhs_axis.plot([], *args, **kwargs)
        self.lines[name] = line

    def legend(self, legend):
        self.axis.legend(legend)

    def update_line(self, name, xdata, ydata):
        self.lines[name].set_data(xdata, ydata)

    def on_draw(self, event):
        if event is not None:
            if event.canvas != self.canvas:
                raise RuntimeError
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)
        self.draw_lines()

    def draw_lines(self):
        for line in self.lines.values():
            self.figure.draw_artist(line)

    def update(self):
        if self.background is None:
            self.on_draw(None)
        elif self.fast:
            self.canvas.restore_region(self.background)
            self.draw_lines()
            self.canvas.blit(self.figure.bbox)
        else:  # slow mode, if blitting doesn't work
            self.canvas.draw()
        self.canvas.flush_events()
