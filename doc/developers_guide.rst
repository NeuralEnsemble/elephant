=================
Developers' guide
=================

These instructions are for developing on a Unix-like platform, e.g. Linux or
Mac OS X, with the bash shell. If you develop on Windows, please get in touch.


Mailing lists
-------------

General discussion of Elephant development takes place in the `NeuralEnsemble Google
group`_.

Discussion of issues specific to a particular ticket in the issue tracker should
take place on the tracker.


Using the issue tracker
-----------------------

If you find a bug in Elephant, please create a new ticket on the `issue tracker`_,
setting the type to "defect".
Choose a name that is as specific as possible to the problem you've found, and
in the description give as much information as you think is necessary to
recreate the problem. The best way to do this is to create the shortest possible
Python script that demonstrates the problem, and attach the file to the ticket.

If you have an idea for an improvement to Elephant, create a ticket with type
"enhancement". If you already have an implementation of the idea, open a pull request.


Requirements
------------

See :doc:`install`. We strongly recommend using virtualenv_ or similar.


Getting the source code
-----------------------

We use the Git version control system. The best way to contribute is through
GitHub_. You will first need a GitHub account, and you should then fork the
repository at https://github.com/NeuralEnsemble/elephant
(see http://help.github.com/fork-a-repo/).

To get a local copy of the repository::

    $ cd /some/directory
    $ git clone git@github.com:<username>/elephant.git
    
Now you need to make sure that the ``elephant`` package is on your PYTHONPATH.
You can do this by installing Elephant::

    $ cd elephant
    $ python setup.py install
    $ python3 setup.py install

but if you do this, you will have to re-run ``setup.py install`` any time you make
changes to the code. A better solution is to install Elephant with the *develop* option,
this avoids reinstalling when there are changes in the code::

    $ python setup.py develop

or::

    $ pip install -e .

To update to the latest version from the repository::

    $ git pull


Running the test suite
----------------------

Before you make any changes, run the test suite to make sure all the tests pass
on your system::

    $ cd elephant/test

With Python 2.7 or 3.x::

    $ python -m unittest discover
    $ python3 -m unittest discover

If you have nose installed::

    $ nosetests

At the end, if you see "OK", then all the tests
passed (or were skipped because certain dependencies are not installed),
otherwise it will report on tests that failed or produced errors.


Writing tests
-------------

You should try to write automated tests for any new code that you add. If you
have found a bug and want to fix it, first write a test that isolates the bug
(and that therefore fails with the existing codebase). Then apply your fix and
check that the test now passes.

To see how well the tests cover the code base, run::

    $ nosetests --with-coverage --cover-package=elephant --cover-erase


Working on the documentation
----------------------------

The documentation is written in `reStructuredText`_, using the `Sphinx`_
documentation system. To build the documentation::

    $ cd elephant/doc
    $ make html
    
Then open `some/directory/elephant/doc/_build/html/index.html` in your browser.
Docstrings should conform to the `NumPy docstring standard`_.

To check that all example code in the documentation is correct, run::

    $ make doctest

To check that all URLs in the documentation are correct, run::

    $ make linkcheck


Committing your changes
-----------------------

Once you are happy with your changes, **run the test suite again to check
that you have not introduced any new bugs**. Then you can commit them to your
local repository::

    $ git commit -m 'informative commit message'
    
If this is your first commit to the project, please add your name and
affiliation/employer to :file:`doc/source/authors.rst`

You can then push your changes to your online repository on GitHub::

    $ git push
    
Once you think your changes are ready to be included in the main Elephant repository,
open a pull request on GitHub (see https://help.github.com/articles/using-pull-requests).


Python 3
--------

Elephant should work with Python 2.7 and Python 3.

So far, we have managed to write code that works with both Python 2 and 3.
Mainly this involves avoiding the ``print`` statement (use ``logging.info``
instead), and putting ``from __future__ import division`` at the beginning of
any file that uses division.

If in doubt, `Porting to Python 3`_ by Lennart Regebro is an excellent resource.

The most important thing to remember is to run tests with at least one version
of Python 2 and at least one version of Python 3. There is generally no problem
in having multiple versions of Python installed on your computer at once: e.g.,
on Ubuntu Python 2 is available as `python` and Python 3 as `python3`, while
on Arch Linux Python 2 is `python2` and Python 3 `python`. See `PEP394`_ for
more on this.


Coding standards and style
--------------------------

All code should conform as much as possible to `PEP 8`_, and should run with
Python 2.7 and 3.2-3.5.


Making a release
----------------

.. TODO: discuss branching/tagging policy.

.. Add a section in /doc/releases/<version>.rst for the release.

First, check that the version string (in :file:`elephant/__init__.py`, :file:`setup.py`,
:file:`doc/conf.py`, and :file:`doc/install.rst`) is correct.

Second, check that the copyright statement (in :file:`LICENCE.txt`, :file:`README.md`, and :file:`doc/conf.py`) is correct.

To build a source package::

    $ python setup.py sdist

To upload the package to `PyPI`_ (if you have the necessary permissions)::

    $ python setup.py sdist upload

.. should we also distribute via software.incf.org

Finally, tag the release in the Git repository and push it::

    $ git tag <version>
    $ git push --tags upstream
    

.. make a release branch



.. _Python: http://www.python.org
.. _nose: http://somethingaboutorange.com/mrl/projects/nose/
.. _neo: http://neuralensemble.org/neo
.. _coverage: http://nedbatchelder.com/code/coverage/
.. _`PEP 8`: http://www.python.org/dev/peps/pep-0008/
.. _`issue tracker`: https://github.com/NeuralEnsemble/elephant/issues
.. _`Porting to Python 3`: http://python3porting.com/
.. _`NeuralEnsemble Google group`: http://groups.google.com/group/neuralensemble
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _Sphinx: http://sphinx.pocoo.org/
.. _numpy: http://www.numpy.org/
.. _quantities: http://pypi.python.org/pypi/quantities
.. _PEP394: http://www.python.org/dev/peps/pep-0394/
.. _PyPI: http://pypi.python.org
.. _GitHub: http://github.com
.. _`NumPy docstring standard`: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _`virtualenv`: https://virtualenv.pypa.io/en/latest/
