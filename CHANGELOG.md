# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [3.0.1] - 2023-12-21

### Fixed

- Deprecation warning from using `logging.warn`
- Make sure stacked MPC matrices have floating-point data type
- Type of inequality vectors in wheeled inverted pendulum MPC

## [3.0.0] - 2023-10-10

### Changed

- Renamed the "cart pole" system (misnomer) to "wheeled inverted pendulum"
- Renamed the library to "qpmpc"

## [2.0.0] - 2023-07-28

### Added

- Base class for exceptions raised by this library
- Documentation checks using ``ruff``
- Exception: ``ProblemDefinitionError``
- MPCProblem: setter for the initial state
- MPCProblem: target state trajectory for stage state cost
- MPCQP class to update cost vectors during execution
- Plan: ``first_input`` getter
- Plan: ``is_empty`` property
- Started ``live_plots`` submodule
- Started ``systems`` submodule

### Changed

- Always add state-input inequalities (even when unfeasible)
- Don't assume a ``Problem`` is fully defined in constructor
- Initial and goal states are now keyword arguments of ``MPCProblem``
- Refactor``build_qp`` into a new ``MPCQP`` class
- Rename ``Problem`` to ``MPCProblem``
- Rename ``Solution`` to ``Plan``
- Rename ``stacked_inputs`` to just ``inputs`` in plans
- Rename ``stacked_states`` to just ``states`` in plans
- Solver keyword argument to ``solve_mpc`` is now mandatory
- Use problem class from ``qpsolvers`` internally
- Warn rather than raise an exception when initial state is unfeasible

## [1.0.0] - 2022-08-12

### Added

- New ``mpc_interface`` alternative in the README

### Changed

- ``solve_qp`` now takes a mandatory ``solver`` keyword argument

### Fixed

- Unit tests for the new ``solver`` keyword argument

## [0.7.0] - 2022-04-03

### Added

- ``sparse`` keyword argument to use with sparse QP solvers
- Usage and examples documentation sections

### Changed

- Only export ``Problem``, ``Solution`` and ``solve_mpc`` module-wide

### Fixed

- Edge case where inputs don't affect the first inequality constraints

## [0.6.0] - 2022-03-30

### Added

- Initial import from [pymanoid](https://github.com/stephane-caron/pymanoid/blob/5158d8902df6265604cec5d790e96f0035575c7a/pymanoid/mpc.py)'s ``mpc.py``.
- Humanoid step example
- Triple integrator example

### Changed

- Extend to time-varying state and input transition matrices

[unreleased]: https://github.com/stephane-caron/qpmpc/compare/v3.0.1...HEAD
[3.0.1]: https://github.com/stephane-caron/qpmpc/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/stephane-caron/qpmpc/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/stephane-caron/qpmpc/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/stephane-caron/qpmpc/compare/v0.7.0...v1.0.0
[0.7.0]: https://github.com/stephane-caron/qpmpc/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/stephane-caron/qpmpc/releases/tag/v0.6.0
