"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: 
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..frame import Frame


class DatasetLoader(ABC):
    """
    Interface that every dataset loader uses.

    Behaves like a read-only sequence of Frames: len(loader) frames, indexable 
    with loader[i], and iterable with 'for frame in loader'.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Number of frames available."""

    @abstractmethod
    def __getitem__(self, index: int) -> Frame:
        """Load and return frame 'index' as a Frame (rgb, depth-in-meters, K)."""

    def __iter__(self) -> Iterator[Frame]:
        # Free iteration for any subclass that implements __len__ and __getitem__.
        for i in range(len(self)):
            yield self[i]
