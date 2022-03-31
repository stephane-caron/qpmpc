# ltv-mpc

[**Installation**](https://github.com/tasts-robots/ltv-mpc#installation)
| [**Documentation**](https://tasts-robots.org/doc/ltv-mpc/)
| [**Example**](https://github.com/tasts-robots/ltv-mpc#example)
| [**Contributing**](CONTRIBUTING.md)

[![build](https://img.shields.io/github/workflow/status/tasts-robots/ltv-mpc/CI)](https://github.com/tasts-robots/ltv-mpc/actions)
[![PyPI version](https://img.shields.io/pypi/v/ltv-mpc)](https://pypi.org/project/ltv-mpc/0.6.0/)
![Status](https://img.shields.io/pypi/status/ltv-mpc)

Linear time-variant (LTV) model predictive control in Python. Solve a quadratic program of the form:

> ![ltv-mpc](https://user-images.githubusercontent.com/1189580/160887387-f5c8e1ae-9ade-48d3-8522-589d9bbef64f.svg)

This module is designed for prototyping. If you need performance, check out one of the related libraries below.

## Installation

```sh
pip install ltv-mpc
```

## Usage

This module defines a one-stop shop function:

```python
solve_mpc(problem: Problem) -> Solution
```

The [``Problem``](https://tasts-robots.org/doc/ltv-mpc/usage.html#ltv_mpc.problem.Problem) type defines the model predictive control problem (LTV system, LTV constraints, initial state and cost function to optimize) while the [``Solution``](https://tasts-robots.org/doc/ltv-mpc/usage.html#ltv_mpc.solution.Solution) holds the resulting state and input trajectories.

## Example

Let us define a triple integrator:

```python
    import numpy as np

    horizon_duration = 1.0
    nb_timesteps = 16
    T = horizon_duration / nb_timesteps
    A = np.array([[1.0, T, T ** 2 / 2.0], [0.0, 1.0, T], [0.0, 0.0, 1.0]])
    B = np.array([T ** 3 / 6.0, T ** 2 / 2.0, T]).reshape((3, 1))
```

Suppose for the sake of example that acceleration is the main constraint acting on our system. We thus define an acceleration constraint ``|acceleration| <= max_accel``:

```python
    max_accel = 3.0  # [m] / [s] / [s]
    accel_from_state = np.array([0.0, 0.0, 1.0])
    ineq_matrix = np.vstack([+accel_from_state, -accel_from_state])
    ineq_vector = np.array([+max_accel, +max_accel])
```

This leads us to the following linear MPC problem:

```python
    from ltv_mpc import Problem

    initial_pos = 0.0
    goal_pos = 1.0
    problem = Problem(
        transition_state_matrix=A,
        transition_input_matrix=B,
        ineq_state_matrix=ineq_matrix,
        ineq_input_matrix=None,
        ineq_vector=ineq_vector,
        initial_state=np.array([initial_pos, 0.0, 0.0]),
        goal_state=np.array([goal_pos, 0.0, 0.0]),
        nb_timesteps=nb_timesteps,
        terminal_cost_weight=1.0,
        stage_state_cost_weight=None,
        stage_input_cost_weight=1e-6,
    )
```

We can solve it with:

```python
    from ltv_mpc import solve_mpc

    solution = solve_mpc(problem, mpc)
```

The solution holds complete state and input trajectories as stacked vectors. For instance, we can plot positions, velocities and accelerations as follows:

```python
    import pylab

    t = np.linspace(0.0, horizon_duration, nb_timesteps + 1)
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

## 🏗️ Work in progress

This module is still under development and its API might change. Future works may include:

- Complete documentation
- Complete test coverage
- General linear stage cost functions

## See also

This module is designed for prototyping. If you need performance, check out one of the following libraries, and [open a PR](https://github.com/tasts-robots/ltv-mpc/pulls) if you know other relevant ones:

| System                | Library                                                  | Language | License      |
|-----------------------|----------------------------------------------------------|----------|--------------|
| Linear time-invariant | [Copra (original)](https://github.com/jrl-umi3218/copra) | C++      | BSD-2-Clause |
| Linear time-variant   | [Copra (fork)](https://github.com/ANYbotics/copra)       | C++      | BSD-2-Clause |
| Nonlinear             | [Crocoddyl](https://github.com/loco-3d/crocoddyl)        | C++      | BSD-3-Clause |
