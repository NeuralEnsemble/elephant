import unittest
from test_online import TestOnlineUnitaryEventAnalysis
from concurrencytest import ConcurrentTestSuite, fork_for_tests

if __name__ == "__main__":

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTest(loader.loadTestsFromTestCase(TestOnlineUnitaryEventAnalysis))
    runner = unittest.TextTestRunner(verbosity=3)

    # runs test sequentially
    # result = runner.run(suite)

    # runs tests across 4 processes
    concurrent_suite = ConcurrentTestSuite(suite, fork_for_tests(4))
    runner.run(concurrent_suite)
