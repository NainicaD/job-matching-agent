"""job-matching-agent: match a consolidated resume profile against job descriptions.

This package is deliberately kept free of top-level imports. Importing
``jobmatch`` (or ``jobmatch.local_matcher``) must never pull in the Anthropic
SDK, so that ``--local`` works on a machine with no ``anthropic`` installed and
no network access. See ``tests/test_local_isolation.py``, which enforces this.
"""

__version__ = "1.0.0"
