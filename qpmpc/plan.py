#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Process solutions to a model predictive control problem."""

import logging
from typing import Optional

import numpy as np
import qpsolvers

from .mpc_problem import MPCProblem


class Plan:
    r"""State and input trajectories that optimize an MPC problem.

    See also the :class:`qpmpc.problem.Problem` class.

    Attributes:
        problem: Model predictive control problem that was solved.
        qpsol: Solution of the corresponding quadratic program.
    """

    __inputs: Optional[np.ndarray]
    __states: Optional[np.ndarray]
    problem: MPCProblem
    qpsol: qpsolvers.Solution

    def __init__(self, problem: MPCProblem, qpsol: qpsolvers.Solution):
        """Test."""
        inputs = None
        if qpsol.found:
            U = qpsol.x
            U = U.reshape((problem.nb_timesteps, problem.input_dim))
            inputs = U
        self.__inputs = inputs
        self.__states = None
        self.problem = problem
        self.qpsol = qpsol

    @property
    def is_empty(self) -> bool:
        """Check whether the plan is empty."""
        return self.__inputs is None

    @property
    def first_input(self) -> Optional[np.ndarray]:
        """Get the first control input of the solution.

        Returns:
            First input if a solution was found, ``None`` otherwise.

        In model predictive control, this is the part of the solution we are
        mainly interested in.
        """
        if self.__inputs is None:
            return None
        return self.__inputs[0]

    @property
    def inputs(self) -> Optional[np.ndarray]:
        r"""Stacked input vector, or ``None`` if the plan is empty.

        This is the stacked vector :math:`U` structured as:

        .. math::

            U = \begin{bmatrix} u_0 \\ u_1 \\ \vdots \\ u_{N-1} \end{bmatrix}

        with :math:`N` the number of timesteps.

        Returns:
            Stacked input vector if the plan is non-empty, ``None`` otherwise.
        """
        return self.__inputs

    @property
    def states(self) -> Optional[np.ndarray]:
        r"""Stacked vector of states.

        This is the vector :math:`X` structured as:

        .. math::

            X = \begin{bmatrix} x_0 \\ x_1 \\ \vdots \\ x_N \end{bmatrix}

        with :math:`N` the number of timesteps.

        Returns:
            Stacked state vector if the plan is non-empty, ``None`` otherwise.

        Note:
            The time complexity of calling this property is :math:`O(N)` the
            first time, then :math:`O(1)` as the result is memoized.
        """
        if self.__inputs is None:  # same as is_empty, helps mypy
            return None
        if self.__states is None:
            x_init = self.problem.initial_state
            if x_init is not None:
                self.__states = self.problem.integrate(x_init, self.__inputs)
            else:  # undefined initial state
                logging.warning("Problem has undefined initial state")
                return None
        return self.__states
