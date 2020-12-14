=============================================
Elephant - Electrophysiology Analysis Toolkit
=============================================

*Elephant* (Electrophysiology Analysis Toolkit) is an emerging open-source,
community centered library for the analysis of electrophysiological data in
the Python programming language.

The focus of Elephant is on generic analysis functions for spike train data and
time series recordings from electrodes, such as the local field potentials
(LFP) or intracellular voltages. In addition to providing a common platform for
analysis codes from different laboratories, the Elephant project aims to
provide a consistent and homogeneous analysis framework that is built on a
modular foundation. Elephant is the direct successor to Neurotools_ and
maintains ties to complementary projects such as ephyviewer_ and
neurotic_ for raw data visualization.

The input-output data format is either Neo_, Quantity_ or Numpy_ array.
Quantity is a Numpy-wrapper package for handling physical quantities like
seconds, milliseconds, Hz, volts, etc. Quantity is used in both Neo and
Elephant.


**Visualization of Elephant analysis objects**

`Viziphant <https://viziphant.readthedocs.io/en/latest/>`_ package is developed
by Elephant team and provides a high-level API to easily generate plots and
interactive visualizations of neuroscientific data and analysis results.
The API uses and extends the same structure as in Elephant to ensure intuitive
usage for scientists that are used to Elephant.


*****************
Table of Contents
*****************

* :doc:`install`
* :doc:`tutorials`
* :doc:`modules`
* :doc:`contribute`
* :doc:`release_notes`
* :doc:`acknowledgments`
* :doc:`authors`
* :doc:`citation`


.. toctree::
    :maxdepth: 2
    :hidden:

    install
    tutorials
    modules
    contribute
    release_notes
    acknowledgments
    authors
    citation


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`


.. _Neurotools:  http://neuralensemble.org/NeuroTools/
.. _ephyviewer:  https://ephyviewer.readthedocs.io/en/latest/
.. _neurotic:  https://neurotic.readthedocs.io/en/latest/
.. _Neo: http://neuralensemble.org/neo/
.. _Numpy: http://www.numpy.org/
.. _Quantity: https://python-quantities.readthedocs.io/en/latest/


.. |date| date::
.. |time| date:: %H:%M
