# Copyright 2020 Optuna, Hugging Face
# License: Apache-2.0

# Copyright 2023 OLMo Authors
# License: Apache-2.0

import functools
import logging
from enum import Enum


@functools.lru_cache(None)
def warning_once(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    self.warning(*args, **kwargs)


logging.Logger.warning_once = warning_once
logging.Logger.warn_once = warning_once


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"
