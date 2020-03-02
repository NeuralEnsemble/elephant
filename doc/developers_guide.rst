.. _developers_guide:

=================
Developers' Guide
=================

.. note:: The documentation guide (how to write a good documentation, naming
          conventions, docstring examples) is in :ref:`documentation_guide`.


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

3. Install requirements.txt, (optionally) requirements-extras.txt, and
   requirements-tests.txt::

    $ pip install -r requirements/requirements.txt
    $ pip install -r requirements/requirements-extras.txt  # optional
    $ pip install -r requirements/requirements-tests.txt

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

6. If it was a new functional, please write:

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


.. note:: If you experience a problem during one of the steps above, please
          contact us by :ref:`get_in_touch`.
