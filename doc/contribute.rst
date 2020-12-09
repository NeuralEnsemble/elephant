.. _contribute:

========================
Contributing to Elephant
========================

You are here to help with Elephant? Awesome, feel welcome to :ref:`get_in_touch`,
where you can ask questions, propose features and improvements to Elephant.

Below is the guide about how to implement a new function in Elephant and come
up with a pull request.

Adding new functionality
************************

1.  Set up a development environment, described in :ref:`developers_guide`
    (sections 1-4).

2.  Write a function you want to add either in one of the existing modules
    (python scripts) or create a new one.

    Imagine that you want to add in existing module ``elephant/statistics.py`` some
    statistics calculation of a list of spike trains that returns an analog signal.
    Let's call it ``statistics_x``.

    Elephant relies on Neo structures that use Quantities extensively, allowing
    the users not to care about the different units (seconds and milliseconds)
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


3.  Write at least one test in ``elephant/test/`` folder that covers the functionality.

    For example, to check the correctness of the implemented ``statistics_x`` function, we add
    unittest code in ``elephant/test/test_statistics.py``, something like

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

4.  Create a pull request, as described in :ref:`developers_guide` (steps 8 and 9).


Done! We'll guide you with the pull request process.
