# ADR 2. Issue and Pull Request Labels
## Context
- Currently the repository has labels shared by issues and PRs.
- Some of the labels have no description and some others have outdated descriptions.


## Options
Status: `Keep` (unchanged), `Update` (new description), `Remove`, `New`.

| Label | Status | Current description | New description |
| ----- | ------ | ------------------- | --------------- |
| `bug` | Update | Indicates an unexpected problem or unintended behavior | Issue: something does not work as documented or expected |
| `bugfix` | Update | Fix for an indentified bug. | PR: fixes a reported bug (link the issue) |
| `enhancement` | Update | Editing an existing module, improving something | Improves an existing function or module (behaviour, API, usability) |
| `new functionality` | Update | New modules, functions | Adds a new scientific method (new function or module) |
| `optimization` | Update |  | Improves speed or memory use without changing results |
| `breaking-changes` | Update |  | Changes public API or results; needs a release-notes entry |
| `documentation` | Keep | Indicates a need for improvements or additions to documentation | |
| `build` | Update | Travis, libraries, package versioning | Packaging, dependencies, versioning and CI configuration |
| `maintenance` | Update |  | Refactoring, code style or cleanup without functional change |
| `Neo related` | Update |  | Concerns Neo data structures or compatibility with Neo |
| `provenance` | Update |  | Concerns provenance tracking of analyses |
| `question` | Update |  | Usage question; consider the Elephant Support room on Matrix |
| `duplicate` | Keep | Indicates similar issues, pull requests, or discussions | |
| `wontfix` | Keep | Indicates that work won't continue on an issue, pull request, or discussion | |
| `help wanted` | Keep | Indicates that a maintainer wants help on an issue or pull request | |
| `build html` | Remove | This will trigger the upload of the docs html files as artifact in github actions | |
| `needs-triage` | New |  | Issue: not yet looked at by a maintainer |
| `needs-discussion` | New |  | Issue: idea or scope not agreed yet |
| `ready-for-PR` | New |  | Issue: agreed by a maintainer, a PR may be opened |
| `in-progress` | New |  | Issue: someone is working on it, see linked PR |

## Decision
Proposed

## Discussion
- Also add `minor` / `major` labels for authoship scoping?
- Any other suggestions
