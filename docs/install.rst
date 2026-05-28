Installation guide
==================

************
Main package
************

The ``storm`` package can be installed from a clone of the repository
using pip:

.. code-block:: bash
    :linenos:

    git clone <repo-url> storm
    cd storm
    pip install -e .

.. note::
    To avoid potential dependency conflicts, installing within a
    `conda environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__
    is recommended.


*********************
Optional dependencies
*********************

Some functions in the ``storm`` package use metacell aggregation via k-Means clustering,
which can receive significant speed up with the `faiss <https://github.com/facebookresearch/faiss>`__ package.

You may install ``faiss`` following the official `guide <https://github.com/facebookresearch/faiss/blob/main/INSTALL.md>`__.

Now you are all set. Proceed to :doc:`tutorials <tutorials>` for how to use the ``storm`` package.
