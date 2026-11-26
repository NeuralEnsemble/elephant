# ADR 1. Documenting Unit Tests
## Context
- [PR #721](https://github.com/NeuralEnsemble/elephant/pull/721): the new tests in `test_datasets.py` use a structured docstring (`Intent` / `Condition` / `Expected`). Elephant has a mixture of `#` comments and docstrings to describe tests, so this introduces a new style.
- Few of the `test_*` functions in `elephant/test/` have a docstring, the rest have a `#` comment block at the top of the body or nothing
- `doc/contribute.rst` says how to add a test, not how to describe it.
- Intrestingly `unittest -v` shows the first docstring line as the test description, `#` comments aren't shown anywhere, but pytest doesn't have this feature by default unless we use a plugin.

Current style (comment):
```python
def test_download_with_cache_dir(self):
    # This test is expected in the GitHub Actions environment where the
    # variable is set to a local cache path. This tests that the function
    # correctly uses the cache directory and returns the expected file
    # path without re-downloading.
```

Style in PR #721 (structured docstring):
```python
def test_valid_data_version_forbidden(self):
    """
    Intent: Ensure only a missing version (404) falls back to master
    Condition: Version probe of `elephant-data` answers 403
    Expected: HTTPError 403 raised, nothing downloaded from master
    """
```
Note: The Intent / Condition / Expected were made up ad hoc while writing the PR #721 tests, without knowing BDD's (Behavious Driven Development) Given / When / Then exists, which is a followed convention.

## Options
|     | Option                                          | Pro                                                                                                   | Con                                                                                   |
| --- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| 1   | Keep as is, some docstring with a descriptive test name                            |                                                                     | 1. Reviewers need to work out what a test checks from its body and/or test name<br>2. Really long test names|
| 2   | Structured docstring `Intent` / `Condition` / `Expected` | 1. States why, the case, and the outcome for each test<br>2. Missing or vague field is easy to spot in review | 1. Maybe overkill for trivial tests<br>2. Own field names, not a known standard |
| 3   | Structured docstring `Given` / `When` / `Then`  | 1. Widely known (BDD, Gherkin)                                                                        | 1. No field for *why* (Intent) the test exists<br>2. Could implies we use BDD tooling |

## Decision
Accepted: Option 2
- Make it a requirment for new tests and tests whose body a PR changes.
- Would only be needed for non-trivial tests, where Intent in not immediately clear.
- `Intent`: why the test exists (behaviour or bug it protects), recommend to start this with a verb like "Ensure", "Reproduce" etc..
- `Condition`: the input or environment that makes this case different from the others in the class
- `Expected`: the observable outcome that the assertions check
- `Context`: (Optional) Longer explaination of the test
- `Tag`: Group unit test, e.g. `type-regression`
- Instructions to add unit-test docstrings added to `doc/contribute.rst` as well.

## Discussion
- Should there be a context field to explain the test in detail.
	- In essense should we limit the fields to one sentence.
	- Decision: Not needed to limit to one sentence, also can have an optional field called "context" to have longer explanations. 
- Field names: keep `Intent` / `Condition` / `Expected`, or use BDD notation with an extra `Intent` line?
	- Decision: Go with Intent, Condition, Expected with an optional Context
- Any other suggestions
	- Would be nice to have a tag to mention something like a regression test
	- Decision: Have one more field called `tag/type` to group unit tests.
