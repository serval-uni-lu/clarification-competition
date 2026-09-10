---
name: Library request
about: Request an additional Python package for use in a competition submission
title: "Library request: <package>"
labels: library-request
---

<!--
Submissions may only import the standard library and the packages already listed
in pyproject.toml. Use this issue to request an additional package. Organizers
review the request and, if approved, add it to pyproject.toml and uv.lock.
Do not include dependency changes in your submission pull request.
-->

## Package

| | |
| --- | --- |
| **Package name** | <!-- as on PyPI --> |
| **Version constraint** | <!-- e.g. >=1.4,<2 --> |
| **License** | <!-- e.g. MIT, Apache-2.0 --> |
| **Homepage / repository** | <!-- link --> |

## Team

<!-- Team name and main contact (as in your submission header) -->

## Why is it needed?

<!-- What does your algorithm use the package for? Why is the standard library or an already-available package (numpy, litellm, ...) not sufficient? -->

## Footprint

- [ ] Pure Python (no compiled extensions or system dependencies)
- [ ] Does not download models, data, or other resources at runtime
- [ ] Does not require network access, subprocesses, or additional services
- [ ] Our algorithm still runs (possibly degraded) without the package until the request is approved

<!-- If any box is unchecked, explain what is required. -->
