# ltv-mpc

[**Installation**](https://github.com/tasts-robots/ltv-mpc#installation)
| [**Usage**](https://github.com/tasts-robots/ltv-mpc#usage)
| [**Example**](https://github.com/tasts-robots/ltv-mpc#example)
| [**Areas of improvement**](https://github.com/tasts-robots/ltv-mpc#areas-of-improvement)
| [**Alternatives**](https://github.com/tasts-robots/ltv-mpc#alternatives)

[![Build](https://img.shields.io/github/workflow/status/tasts-robots/ltv-mpc/CI)](https://github.com/tasts-robots/ltv-mpc/actions)
[![Coverage](https://coveralls.io/repos/github/tasts-robots/ltv-mpc/badge.svg?branch=master)](https://coveralls.io/github/tasts-robots/ltv-mpc?branch=master)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen?logo=read-the-docs&style=flat)](https://tasts-robots.org/doc/ltv-mpc/)
[![PyPI version](https://img.shields.io/pypi/v/ltv-mpc)](https://pypi.org/project/ltv-mpc/0.6.0/)
![Status](https://img.shields.io/pypi/status/ltv-mpc)

Linear time-variant (LTV) model predictive control in Python. Solve a quadratic program of the form:

> ![ltv-mpc](https://raw.githubusercontent.com/tasts-robots/ltv-mpc/master/doc/src/images/ltv-mpc.svg)

This module is designed for prototyping. If you need performance, check out the [alternatives](#alternatives) below.

📢 **2022-08:** the brand new [mpc_interface](https://github.com/Gepetto/mpc-interface) handles more general cost functions.

## Installation

```sh
pip install ltv-mpc
```

## Usage

This module defines a one-stop shop function:

```python
solve_mpc(problem: Problem, solver: str) -> Solution
```

The [``Problem``](https://tasts-robots.org/doc/ltv-mpc/usage.html#ltv_mpc.problem.Problem) type defines the model predictive control problem (LTV system, LTV constraints, initial state and cost function to optimize) while the [``Solution``](https://tasts-robots.org/doc/ltv-mpc/usage.html#ltv_mpc.solution.Solution) holds the resulting state and input trajectories. The ``solver`` string is used to select the backend [quadratic programming solver](https://github.com/stephane-caron/qpsolvers#solvers).

## Example

Let us define a triple integrator:

```python
    import numpy as np

    horizon_duration = 1.0  # [s]
    N = 16  # number of discretization steps
    T = horizon_duration / N
    A = np.array([[1.0, T, T ** 2 / 2.0], [0.0, 1.0, T], [0.0, 0.0, 1.0]])
    B = np.array([T ** 3 / 6.0, T ** 2 / 2.0, T]).reshape((3, 1))
```

Suppose for the sake of example that acceleration is the main constraint acting on our system. We thus define an acceleration constraint ``|acceleration| <= max_accel``:

```python
    max_accel = 3.0  # [m] / [s] / [s]
    accel_from_state = np.array([0.0, 0.0, 1.0])
    C = np.vstack([+accel_from_state, -accel_from_state])
    e = np.array([+max_accel, +max_accel])
```

This leads us to the following linear MPC problem:

```python
    from ltv_mpc import Problem

    x_init = np.array([0.0, 0.0, 0.0])
    x_goal = np.array([1.0, 0.0, 0.0])
    problem = Problem(
        transition_state_matrix=A,
        transition_input_matrix=B,
        ineq_state_matrix=C,
        ineq_input_matrix=None,
        ineq_vector=e,
        initial_state=x_init,
        goal_state=x_goal,
        nb_timesteps=N,
        terminal_cost_weight=1.0,
        stage_state_cost_weight=None,
        stage_input_cost_weight=1e-6,
    )
```

We can solve it with:

```python
    from ltv_mpc import solve_mpc

    solution = solve_mpc(problem, solver="quadprog")
```

The solution holds complete state and input trajectories as stacked vectors. For instance, we can plot positions, velocities and accelerations as follows:

```python
    import pylab

    t = np.linspace(0.0, horizon_duration, N + 1)
    X = solution.stacked_states
    positions, velocities, accelerations = X[:, 0], X[:, 1], X[:, 2]
    pylab.ion()
    pylab.plot(t, positions)
    pylab.plot(t, velocities)
    pylab.plot(t, accelerations)
    pylab.grid(True)
    pylab.legend(("position", "velocity", "acceleration"))
```

This example produces the following trajectory:

![2022-03-30-172206_1920x1080_scrot](https://user-images.githubusercontent.com/1189580/160871543-3734ec65-fe74-4a6f-8452-a877aa4050b1.png)

The behavior is a weighted compromise between reaching the goal state (weight ``1.0``) and keeping reasonable finite jerk inputs (weight ``1e-6``). The latter mitigate bang-bang accelerations but prevent fully reaching the goal within the horizon. See the [examples](examples/) folder for more examples.

## Areas of improvement

This module is incomplete with regards to the following points:

- Cost functions: can be extended to general linear stage cost functions
- Documentation: there are some undocumented functions
- Test coverage: only one end-to-end test

New [contributions](CONTRIBUTING.md) are welcome :)

## Alternatives

This module is designed for prototyping. If you need performance, check out one of the following libraries, and [open a PR](https://github.com/tasts-robots/ltv-mpc/pulls) if you know other relevant ones:

| Library                                                    | System                | Language   | License      |
|------------------------------------------------------------|-----------------------|------------|--------------|
| [Copra (original)](https://github.com/jrl-umi3218/copra)   | Linear time-invariant | C++        | BSD-2-Clause |
| [Copra (fork)](https://github.com/ANYbotics/copra)         | Linear time-variant   | C++        | BSD-2-Clause |
| [mpc\_interface](https://github.com/Gepetto/mpc-interface) | Linear time-variant   | C++/Python | BSD-2-Clause |
| [Crocoddyl](https://github.com/loco-3d/crocoddyl)          | Nonlinear             | C++        | BSD-3-Clause |
