# ADR 3. Contribution Policy

## Context
- More and more AI-generated issues and PRs reach Elephant, many of them low-effort.
- Each one still costs maintainer time to read and review.
- The previous contributor guide (`doc/contribute.rst`) had:
	- no policy on AI-assisted contributions,
	- no requirement to open or discuss an issue before a PR,
	- no statement of what a PR must contain to be reviewed,
	- no statement of what happens to PRs that do not meet the bar,
	- automatic authorship for any PR (contributors were asked to add themselves to `doc/authors.rst`).
- Goal: raise the bar without discouraging genuine (incl. first-time) contributors; make expectations explicit and automate checks (templates, CI) where possible.
- How other projects handle it:
	- [numpy](https://numpy.org/contribute/): cap on open PRs for non-maintainers.
	- [scipy](https://scipy.github.io/devdocs/dev/conduct/ai_policy.html): a human must be in the loop.
	- [pandas](https://pandas.pydata.org/docs/dev/development/contributing.html): disclose tool/model and what it was used for; must not use AI to speak for you; self-review skill for contributors.
	- [mne](https://github.com/mne-tools/mne-python/blob/main/CONTRIBUTING.md): maintainers may close fully auto-generated PRs.
	- [matplotlib](https://matplotlib.org/devdocs/devel/contribute.html#use-of-generative-ai): engage deeply with one or two projects instead of many.
	- [linux kernel](https://github.com/torvalds/linux): separate instructions for coding agents.
	- [numpy changelog.py](https://github.com/numpy/numpy/blob/main/tools/changelog.py): tiered recognition, contributor lists generated from git log.

## Accepted

### AI-assisted contributions
- Human in the loop: the contributor understands and can explain every line they submit.
- Disclosure: the PR body names the tool/model and what it was used for.
- No AI-written replies in issue or PR discussions.
- Maintainers may close fully automated PRs without detailed review.
- Engage deeply with one project rather than many; PRs should be meaningful contributions.
- Self-review checklist in the guide that contributors go through before opening a PR.

### Communication
- Flow: Matrix chat (optional) -> issue -> PR.
- Matrix rooms: Elephant Support (usage questions), Elephant Feature Requests (in-depth discussion before an issue).
- The mailing list is kept.

### Before opening a PR
- Open or link an issue first and wait for maintainer agreement.
- Trivial changes are exempt, defined with examples: typos, broken links, small docstring fixes.
- At most 2 open PRs per non-maintainer contributor.

### PR requirements
- PR template with required fields: linked issue, what/why, how it was tested, AI disclosure.
- Checkboxes: issue approved by a maintainer (or trivial), fast CI passes, contributor can explain all changes.
- Fast CI must pass before a review is requested (not for first-time contributors, whose CI needs maintainer approval).
- One topic per PR, "as large as necessary, as small as possible".

### After opening a PR
- The guide states what we review and roughly how long it may take.
- PRs not meeting the requirements may be closed without review; they can be reopened once they do.
- Stale PRs: closed after 8 weeks without a response to a maintainer's question or change request.

### Recognition
- Two tiers instead of three.
- All contributors: credited in the release notes, list generated from git log (incl. `Co-authored-by:`), first-timers marked.
- `doc/authors.rst`: "Maintainers will invite you to add yourself to the authors list."; nobody adds themselves unprompted.
- Contributors can opt out of being listed in the release notes.

### Other welcome contributions
- Listed in the guide: bug reports, documentation, tutorials, testing PRs.

## Rejected
- `ai-fixable` label: would attract more bots.
- Release-notes entry written by the contributor.
- Reviews of PRs as a contribution, including counting them towards authorship.
- Zenodo creators / citation authors process: out of scope.

## Discussion
- Agent instructions in the repo: should we have an `AGENTS.md` and/or a `CLAUDE.md`?
- Location of `CONTRIBUTING.md`: `.github/` or repo root?
	- Neither location is shipped in the sdist or wheel.
	- Note: this may change later, e.g. if the docs switch to a Sphinx build with Markdown sources, `CONTRIBUTING.md` could be used directly as the guide.
- What should the saved replies / standard closing messages for low-effort PRs say?
- Should the `doc/contribute.rst` list counter-examples of what is not trivial, even if the change is small?
- Should `doc/contribute.rst` state the criteria for when maintainers invite you to the authors list, or leave it open-ended?
	- Draft criteria, at least one of:
		- contributed a new function, class, or module that was merged into Elephant;
		- substantially extended or improved existing functionality (e.g. a new method option, a major performance improvement, or a rewrite);
		- made sustained contributions over time, e.g. several merged bug fixes, documentation, or tutorial improvements across releases;
		- provided substantial scientific input, e.g. the method that a module implements.
- Should the opt-out sentence (`doc/contribute.rst`, "Release notes: all contributors") also cover the authors list?
	- Currently: "If you do not want to be listed in the release notes, say so in the description of your pull request."
	- Alternative: "If you do not want to be listed in the release notes or the authors list, say so in the description of your pull request."
- Wording of the intro above the list in `doc/authors.rst`?
	- Currently: "The following people have contributed code and/or ideas to the current version of Elephant...": does it need new wording now that maintainers invite people?
- Any other suggestions
