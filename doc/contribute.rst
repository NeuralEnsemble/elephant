.. _contribute:

========================
Contributing to Elephant
========================

You are here to help with Elephant? Awesome, feel welcome to get in touch with
us by asking questions, proposing features and improvements to Elephant.

For guidelines on code documentation, please see the :ref:`documentation_guide`
below.


.. note::

    We highly recommend to get in touch with us *before* starting to implement a
    new feature in Elephant. This way, we can point out synergies with complementary
    efforts and help in designing your implementation such that its integration
    into Elephant will be an easy process.


.. _get_in_touch:

************
Get in touch
************

Using the mailing list
----------------------

General discussion of Elephant development takes place in the
`NeuralEnsemble Google group <http://groups.google.com/group/neuralensemble>`_.

Please raise concrete issues, bugs, and feature requests on the `issue tracker`_.


Using the issue tracker
-----------------------

If you find a bug in Elephant, please create a new ticket on the
`issue tracker`_. Choose one of the available templates - "Bug report",
"Feature request", or "Questions".
Choose an issue title that is as specific as possible to the problem you've found, and
in the description give as much information as you think is necessary to
recreate the problem. The best way to do this is to create the shortest possible
Python script that demonstrates the problem, and attach the file to the ticket.

If you have an idea for an improvement to Elephant, create a ticket with type
"enhancement". If you already have an implementation of the idea, open a
`pull request <https://github.com/NeuralEnsemble/elephant/pulls>`_.

.. _set_up_an_environment:

************************************
Setting up a development environment
************************************

In order to contribute to the Elephant code, you must set up a Python environment on
your local machine.

1. Fork `Elephant <https://github.com/NeuralEnsemble/elephant>`_ as described
   in `Fork a repo <https://help.github.com/en/github/getting-started-with-github/fork-a-repo>`_.
   Download Elephant source code from your forked repo::

    $ git clone git://github.com/<your-github-profile>/elephant.git
    $ cd elephant

2. Set up the virtual environment (either via pip or conda):

.. tabs::

    .. tab:: conda (recommended)

        .. code-block:: sh

            conda env create -f requirements/environment.yml
            conda activate elephant
            pip install -r requirements/requirements-tests.txt

    .. tab:: pip

        .. code-block:: sh

            python -m venv elephant-env
            source elephant-env/bin/activate

            pip install -r requirements/requirements.txt
            pip install -r requirements/requirements-extras.txt  # optional
            pip install -r requirements/requirements-tests.txt


3. Before you make any changes, run the test suite to make sure all the tests
   pass on your system::

    $ nosetests .

   You can specify a particular module to test, for example
   ``test_statistics.py``::

    $ nosetests elephant/test/test_statistics.py

   At the end, if you see "OK", then all the tests passed (or were skipped
   because certain dependencies are not installed), otherwise it will report
   on tests that failed or produced errors.


************************
Adding new functionality
************************

After you've set up a development environment, implement a functionality you
want to add in Elephant. This includes, in particular:

   * fixing a bug;
   * improving the documentation;
   * adding a new functionality.

Writing code
------------

Imagine that you want to add a novel
statistical measure of a list of spike trains that returns an analog signal in the existing module ``elephant/statistics.py``.
Let's call it ``statistics_x``.

Elephant relies on Neo structures that use Quantities extensively, allowing
the users to conveniently specify physical units (e.g., seconds and milliseconds)
of input objects. Typically, to avoid computationally expensive quantities
rescaling operation on large input arrays, we check that the main data objects
- spike trains or analog signals - share the same units and rescale additional
parameters (t_start, t_stop, bin_size, etc.) to the units of input objects.

.. code-block:: python

    import neo
    import quantities as pq

    from elephant.utils import check_same_units


    def statistics_x(spiketrains, t_start=None, t_stop=None):
        """
        Compute the X statistics of spike trains.

        Parameters
        ----------
        spiketrains : list of neo.SpikeTrain
            Input spike trains.
        t_start, t_stop : pq.Quantity or None
            Start and stop times to compute the statistics over the specified
            interval. If None, extracted from the input spike trains.

        Returns
        -------
        signal : neo.AnalogSignal
            The X statistics of input spike trains.
            (More description follows.)

        """
        check_same_units(spiketrains, object_type=neo.SpikeTrain)

        # alternatively, if spiketrains are required to be aligned in time,
        # when t_start and t_stop are not specified, use 'check_neo_consistency'
        # check_neo_consistency(spiketrains, object_type=neo.SpikeTrain, t_start=t_start, t_stop=t_stop)

        # convert everything to spiketrain units and strip off the units
        if t_start is None:
            t_start = spiketrains[0].t_start
        if t_stop is None:
            t_stop = spiketrains[0].t_stop
        units = spiketrains[0].units
        t_start = t_start.rescale(units).item()
        t_stop = t_stop.rescale(units).item()
        spiketrains = [spiketrain.magnitude for spiketrain in spiketrains]

        # do the analysis here on unit-less spike train arrays
        x = ...

        signal = neo.AnalogSignal(x,
                                  units=...,
                                  t_start=t_start,
                                  sampling_rate=...,
                                  name="X statistics of spiketrains",
                                  ...)
        return signal


Testing code
------------

Write at least one test in ``elephant/test/test_module_name.py`` file that
covers the functionality.

For example, to check the correctness of the implemented ``statistics_x``
function, we add unittest code in ``elephant/test/test_statistics.py``,
something like

.. code-block:: python

    import unittest

    import neo
    import quantities as pq
    from numpy.testing import assert_array_almost_equal

    from elephant.statistics import statistics_x


    class StatisticsXTestCase(unittest.TestCase):
        def test_statistics_x_correctness(self):
            spiketrain1 = neo.SpikeTrain([0.3, 4.5, 7.8], t_stop=10, units='s')
            spiketrain2 = neo.SpikeTrain([2.4, 5.6], t_stop=10, units='s')
            result = statistics_x([spiketrain1, spiketrain2])
            self.assertIsInstance(result, neo.AnalogSignal)
            self.assertEqual(result.t_start, 0 * pq.s)
            expected_magnitude = [0, 1, 2]
            assert_array_almost_equal(result.magnitude, expected_magnitude)
            ...  # more checking


Pushing the changes and creating a pull request
-----------------------------------------------

Now you're ready to share the code publicly.

1.  Commit your changes:

    .. code-block:: sh

        git add .
        git commit -m "informative commit message"
        git push

    If this is your first commitment to Elephant, please add your name and
    affiliation/employer in :file:`doc/authors.rst`

2.  Open a `pull request <https://github.com/NeuralEnsemble/elephant/pulls>`_.
    Then we will guide you through the process of merging your code into Elephant.

That's all! We're happy to assist you throughout the process of contributing.

If you experience any problems during one of the steps below, please contact us
and we'll help you.


.. _documentation_guide:

*******************
Documentation Guide
*******************


Writing documentation
---------------------

Each module (python source file) should start with a short description of the
listed functionality. Class and function docstrings should conform to the
`NumPy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

.. note:: Please refer to our :doc:`style_guide`.


Building documentation
----------------------

The documentation in :file:`doc/` folder is written in `reStructuredText
<http://docutils.sourceforge.net/rst.html>`_, using the
`Sphinx <http://sphinx-doc.org/>`_ documentation system. To build the
documentation locally on a Linux system, follow these steps:

1. Install requirements-docs.txt and requirements-tutorials.txt the same way
   it's explained in :ref:`set_up_an_environment` step 3:

   .. code-block:: sh

        pip install -r requirements/requirements-docs.txt
        pip install -r requirements/requirements-tutorials.txt

2. Build the documentation:


   .. code-block:: sh

        cd doc
        export PYTHONPATH=${PYTHONPATH}.:../..
        make html

   ``PYTHONPATH`` environmental variable is set in order to find Elephant
   package while executing jupyter notebooks that are part of the documentation.
   You may also need to install LaTeX support:

   .. code-block:: sh

        sudo apt-get install texlive-full

3. Open :file:`_build/html/index.html` in your browser.

4. (Optional) To check that all URLs in the documentation are correct, run:

   .. code-block:: sh

        make linkcheck


Citations
---------

The citations are in BibTeX format, stored in `doc/bib/elephant.bib
<https://github.com/NeuralEnsemble/elephant/blob/master/doc/bib/elephant.bib>`_.

To cite Elephant, refer to :doc:`citation`.

Each module in ``doc/reference`` folder ends with the reference section:

.. code-block:: rst

    References
    ----------

    .. bibliography:: ../bib/elephant.bib
       :labelprefix: <module name shortcut>
       :keyprefix: <module name>-
       :style: unsrt

where ``<module name>`` is (by convention) the Python source file name, and
``<module name shortcut>`` is what will be displayed to the users.

For example, ``:cite:'spade-Torre2013_132'`` will be rendered as ``sp1`` in
the built HTML documentation, if ``<module name shortcut>`` is set to ``sp``
and ``<module name>`` - to ``spade``.

.. _Issue tracker: https://github.com/NeuralEnsemble/elephant/issues
