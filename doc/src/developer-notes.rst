:github_url: https://github.com/tasts-robots/ltv-mpc/tree/main/doc/src/developer-notes.rst

***************
Developer notes
***************

Quadratic program
=================

Internally, :func:`.solve_mpc` builds a quadratic program before calling a QP solver. You can retrieve the QP corresponding to the input problem by calling the internal :func:`.build_mpc_qp` function:

.. autofunction:: ltv_mpc.solve_mpc.build_mpc_qp
