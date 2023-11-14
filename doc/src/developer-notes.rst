:github_url: https://github.com/stephane-caron/qpmpc/tree/main/doc/src/developer-notes.rst

***************
Developer notes
***************

Quadratic program
=================

Internally, :func:`.solve_mpc` builds a quadratic program before calling a QP solver. You can retrieve the QP corresponding to the input problem by creating an instance of the intermediate :func:`.MPCQP` representation:

.. autoclass:: qpmpc.mpc_qp.MPCQP
