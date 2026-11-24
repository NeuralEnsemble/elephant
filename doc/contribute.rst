.. _contribute:

========================
Contributing to Elephant
========================

You are here to help with Elephant? Awesome, you are very welcome! This guide
describes how contributions to Elephant work, from the first idea to a merged
pull request.

For guidelines on code documentation, please see the :ref:`documentation_guide`
below.


.. important::

    **Open an issue first.** Every pull request must be linked to an issue on
    the `issue tracker`_ that a maintainer has agreed to. Please wait for this
    agreement before you open a pull request. Only :ref:`trivial changes
    <trivial_changes>` are exempt from this rule.

    This way, we can point out synergies with complementary efforts, agree on
    the scope and design early, and make sure your work can be integrated into
    Elephant without wasted effort on either side.


.. _contribution_flow:

*****************
Contribution flow
*****************

A contribution to Elephant follows these steps:

1. **Discuss (optional).** If you would like to talk through an idea in depth
   before writing it up, join the `Elephant Feature Requests`_ room on Matrix
   and talk to the maintainers.
2. **Open an issue.** Describe the bug, feature, or improvement on the
   `issue tracker`_ (see :ref:`using_the_issue_tracker`) and wait until a
   maintainer agrees to it.
3. **Open a pull request.** Once the issue is agreed on, implement the change
   and open a pull request that links to the issue (see
   :ref:`set_up_an_environment` and the sections below). If you use AI tools
   for any part of your contribution, follow the rules in
   :ref:`ai_assisted_contributions`.


.. _using_the_issue_tracker:

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
"enhancement".


.. _trivial_changes:

What counts as trivial
----------------------

Small, obviously correct changes may be submitted as a pull request directly,
without an issue. Examples:

* fixing typos in the documentation or in code comments,
* fixing broken links,
* small corrections to docstrings (e.g., a wrong parameter name or type).

If you are unsure whether a change is trivial, it is not: please open an issue.


Other ways to contribute
------------------------

Contributing to Elephant is not limited to code. The following contributions
are equally welcome:

* **Bug reports**: clear, reproducible reports on the `issue tracker`_.
* **Documentation**: improvements to the user guide, the API reference.
* **Tutorials**: new or improved examples showing how to use Elephant.
* **Testing pull requests**: trying out open pull requests and reporting
  whether they work for you.


.. _ai_assisted_contributions:

*************************
AI-assisted contributions
*************************

You may use AI tools (e.g., code assistants or coding agents) when
contributing to Elephant. You remain responsible for everything you submit,
and the following rules apply:

* **Keep a human in the loop.** You must understand every line you submit and
  be able to explain it in review. "The AI wrote it" is not an answer to a
  review question.
* **Disclose AI use.** State in the pull request description which tool or
  model you used and what you used it for (e.g., generating tests, drafting
  docstrings, refactoring).
* **Write your own replies.** Discussions on issues and pull requests are
  between people. Do not post AI-generated comments or replies; if you use a
  tool to help with wording or translation, the content and the reasoning must
  be your own.
* **Fully automated pull requests may be closed.** Maintainers may close pull
  requests that appear to be generated without meaningful human involvement,
  without a detailed review.
* **Engage deeply with one project rather than many.** We value contributors
  who take the time to understand Elephant and its scientific background.
  Pull requests should be meaningful contributions, not mass-produced changes.

Before submitting, go through the :ref:`self-review checklist <self_review_checklist>` yourself.


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

Install elephant development version:

.. code-block:: sh

    pip install -e .

3. Before you make any changes, run the test suite to make sure all the tests
   pass on your system::

    $ pytest .

   You can specify a particular module to test, for example
   ``test_statistics.py``::

    $ pytest elephant/test/test_statistics.py

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

Describing tests
~~~~~~~~~~~~~~~~

New tests, and existing tests whose body your pull request changes, should
have a structured docstring if what they check is not obvious from the name
and a quick look at the body, something like:

.. code-block:: python

    def test_valid_data_version_forbidden(self):
        """
        Intent: Ensure only a missing version (404) falls back to master
        Condition: Version probe of `elephant-data` answers 403
        Expected: HTTPError 403 raised, nothing downloaded from master
        """
Trivial tests can go without one.

Fields:

* ``Intent``: why the test exists, i.e. the behaviour or bug it protects.
  Start it with a verb, e.g. "Ensure ..." or "Reproduce ...".
* ``Condition``: the input or environment that makes this case different from
  the other tests in the class.
* ``Expected``: the observable outcome that the assertions check.
* ``Context`` (optional): a longer explanation of the test, e.g. background on
  the bug or why the setup looks the way it does.
* ``Tag`` (optional): a label to group tests, e.g. ``type-regression``.

Fields are not limited to one sentence; indent continuation lines to align
with the text of the first line.


.. _self_review_checklist:

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

If you experience any problems during one of the steps above, please
:ref:`get in touch <get_in_touch>` and we'll help you.


.. _get_in_touch:

************
Get in touch
************

Besides contributions, you are welcome to contact us with general questions,
feedback, or to discuss the use of Elephant through the following channels:

* The `Elephant Support`_ room on Matrix for questions about using Elephant
  that are not suitable for an issue.
* The `NeuralEnsemble Google group <http://groups.google.com/group/neuralensemble>`_
  mailing list for general discussion of Elephant development.


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

Entries in the .bib are built according to the following specification:

``{Last name of first author}+{last two digits of publication year}+"_"+{first page of article}``


.. admonition:: Example

    According to this specification, the following article:

    S. Grün. Data-driven significance estimation of precise spike correlation. J. Neurophysiol., pages 1126–1140, 2009. doi:10.1152/jn.00093.2008.

    is added to the .bib-file as ``Gruen09_1126``. The complete entry for this example is:

    .. code-block:: rst

        @article{Gruen09_1126,
          title={Data-driven significance estimation of precise spike correlation},
          author={Gr\"{u}n, S.},
          year={2009},
          journal={J. Neurophysiol.},
          number={101},
          pages={1126--1140},
          doi={10.1152/jn.00093.2008}
        }


Each module in ``doc/reference`` folder ends with the reference section:

.. code-block:: rst

    References
    ----------

    .. bibliography::
        :keyprefix: <module name>-

where ``<module name>`` is (by convention) the Python source file name, and
``<module name shortcut>`` is what will be displayed to the users.

For example, ``:cite:'spade-Torre2013_132'`` will be rendered as ``sp1`` in
the built HTML documentation, if ``<module name shortcut>`` is set to ``sp``
and ``<module name>`` - to ``spade``.

To cite Elephant itself, refer to :doc:`citation`.

.. _Issue tracker: https://github.com/NeuralEnsemble/elephant/issues
.. TODO: replace the placeholders below with the room aliases once they exist.
.. _Elephant Support: https://matrix.to/#/TODO-elephant-support
.. _Elephant Feature Requests: https://matrix.to/#/TODO-elephant-feature-requests
