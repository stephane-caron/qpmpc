:github_url: https://github.com/tasts-robots/ltv-mpc/tree/main/doc/src/index.rst

.. title:: Table of Contents

#######
ltv-mpc
#######

Linear time-variant (LTV) model predictive control in Python. Solve a quadratic
program of the form:

.. math::

    \begin{array}{rl}
    \underset{x_k, u_k}{\min} \quad & w_{t} \|x_N - x_{\mathit{goal}}\|_2^2 + w_x \sum_{k=0}^{N-1} \| x_k - x_{\mathit{goal}} \|_2^2  + w_u \sum_{k=0}^{N-1} \| u_k \|^2_2 \\
    \mathrm{s.t.} \quad & x_{k+1} = A_k x_k + B_k u_k \\
    & C_k x_k + D_k u_k \leq e_k \\
    & x_0 = x_{\mathit{init}}
    \end{array}

The library provides a one-stop shop :func:`.solve_mpc` function that computes the :class:`.Solution` corresponding to such a :class:`.Problem`.

.. toctree::

    installation.rst
    usage.rst
    examples.rst
    developer-notes.rst
