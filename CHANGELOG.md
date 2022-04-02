# Changelog

All notable changes to this project will be documented in this file.

## [0.6.1] - 2022/04/03

### Added

- ``sparse`` keyword argument to use with sparse QP solvers
- Usage and examples documentation sections

### Changed

- Only export ``Problem``, ``Solution`` and ``solve_mpc`` module-wide

### Fixed

- Edge case where inputs don't affect the first inequality constraints

## [0.6.0] - 2022/03/30

### Added

- Initial import from [pymanoid](https://github.com/stephane-caron/pymanoid/blob/5158d8902df6265604cec5d790e96f0035575c7a/pymanoid/mpc.py)'s ``mpc.py``.
- Humanoid stepping example
- Triple integrator example

### Changed

- Extend to time-varying state and input transition matrices
