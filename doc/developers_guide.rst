.. _developers_guide:

=================
Developers' Guide
=================

.. note::
    1. The documentation guide (how to write a good documentation, naming
       conventions, docstring examples) is in the :ref:`documentation_guide`.

    2. We highly recommend to get in touch with us (see :ref:`get_in_touch`) *before* starting
       to implement a new feature in Elephant. This way, we can point out synergies with
       complementary efforts and help in designing your implementation such that its integration
       into Elephant will be an easy process.

    3. If you experience any problems during one of the steps below, please
       contact us and we'll help you.


1. Follow the instructions in :ref:`prerequisites` to setup a clean conda
   environment. To be safe, run::

    $ pip uninstall elephant

   to uninstall ``elephant`` in case you've installed it previously as a pip
   package.

2. Fork `Elephant <https://github.com/NeuralEnsemble/elephant>`_ as described
   in `Fork a repo <https://help.github.com/en/github/getting-started-with-github/fork-a-repo>`_.
   Download Elephant source code from your forked repo::

    $ git clone git://github.com/<your-github-profile>/elephant.git
    $ cd elephant

3. Install the requirements (either via pip or conda):

.. tabs::

    .. tab:: pip

        .. code-block:: sh

            pip install -r requirements/requirements.txt
            pip install -r requirements/requirements-extras.txt  # optional
            pip install -r requirements/requirements-tests.txt

        If you install extras, make sure you've installed OpenMPI
        (e.g., for Debian based distributions ``sudo apt install -y libopenmpi-dev openmpi-bin``).

    .. tab:: conda

        If you don't have or don't want to install OpenMPI system-wide,
        use conda.

        .. code-block:: sh

            conda env create -f requirements/environment.yml
            conda activate elephant
            pip install -r requirements/requirements-tests.txt


4. Before you make any changes, run the test suite to make sure all the tests
   pass on your system::

    $ nosetests .

   You can specify a particular module to test, for example
   ``test_statistics.py``::

    $ nosetests elephant/test/test_statistics.py

   At the end, if you see "OK", then all the tests passed (or were skipped
   because certain dependencies are not installed), otherwise it will report
   on tests that failed or produced errors.

5. **Implement the functional you want to add in Elephant**. This includes
   (either of them):

   * fixing a bug;
   * improving the documentation;
   * adding a new functional.

6. If it's a new functional, please write:

   - documentation (refer to :ref:`documentation_guide`);
   - tests to cover your new functions as much as possible.

7. Run the tests again as described in step 4.

8. Commit your changes::

    $ git add .
    $ git commit -m "informative commit message"
    $ git push

   If this is your first commit to the project, please add your name and
   affiliation/employer to :file:`doc/authors.rst`

9. Open a `pull request <https://github.com/NeuralEnsemble/elephant/pulls>`_.
   Then we'll merge your code in Elephant.
