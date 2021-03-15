import os
import inspect
import elephant.buffalo as buffalo


TEST_PROV = os.environ.get('TEST_PROV', False)


def _provenance_test(active=TEST_PROV):
    if active:
        buffalo.provenance.Provenance.set_calling_frame(
            inspect.currentframe().f_back)
        buffalo.provenance.Provenance.active = True
        print("Testing provenance")
