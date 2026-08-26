"""Teacher models that produce distillation targets."""

from .base import (
    ION_COLUMNS,
    PrecursorLabels,
    Teacher,
    labels_from_frames,
    labels_to_frames,
)
from .fake import FakeTeacher

__all__ = [
    "Teacher",
    "PrecursorLabels",
    "ION_COLUMNS",
    "labels_to_frames",
    "labels_from_frames",
    "FakeTeacher",
    "get_teacher",
]


def get_teacher(name: str, **kwargs) -> Teacher:
    """Factory. ``fake`` is always available; ``alphapeptdeep`` needs the teacher extra."""
    if name == "fake":
        return FakeTeacher()
    if name in ("alphapeptdeep", "peptdeep"):
        from .peptdeep_teacher import PeptDeepTeacher

        return PeptDeepTeacher(**kwargs)
    raise ValueError(f"unknown teacher {name!r}; choose from: fake, alphapeptdeep")
