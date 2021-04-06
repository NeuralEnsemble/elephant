import os
import inspect
import elephant.buffalo as buffalo


TEST_PROV = bool(os.environ.get('TEST_PROV', 'false')
                 in ['true', '1']) and buffalo.HAVE_PROV


def _provenance_test(active=TEST_PROV):
    if active:
        buffalo.Provenance.set_calling_frame(inspect.currentframe().f_back)
        buffalo.Provenance.active = True
        print("Testing provenance")
