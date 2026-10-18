"""Compatibility entry point; settings and implementation live in apoapsis_brain.

Use ``python -m scripts.apoapsis_brain check|train|evaluate`` for new runs.
"""

if __package__:
    from .apoapsis_brain import *  # noqa: F403
else:
    from apoapsis_brain import *  # noqa: F403


if __name__ == "__main__":
    main()  # noqa: F405
