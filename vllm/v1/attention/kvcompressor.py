# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV Cache Compressor registry and abstract base class."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager
)

from vllm.v1.core.kv_cache_manager import (
    KVCacheManager
)


if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


# =============================================================================
# Enum Registry
# =============================================================================


class _KVCompressorEnumMeta(EnumMeta):
    """Metaclass for KVCompressorEnum to provide better error messages."""

    def __getitem__(cls, name: str):
        """Get compressor by name with helpful error messages."""
        try:
            return super().__getitem__(name)
        except KeyError:
            members = cast("dict[str, Enum]", cls.__members__).keys()
            valid_compressors = ", ".join(members)
            raise ValueError(
                f"Unknown KV compressor: '{name}'. "
                f"Valid options are: {valid_compressors}"
            ) from None


class KVCompressorEnum(Enum, metaclass=_KVCompressorEnumMeta):
    """Enumeration of all supported KV cache compressors.

    The enum value is the default class path, but this can be overridden
    at runtime using register_compressor().

    To get the actual compressor class (respecting overrides), use:
        compressor.get_class()
    """

    # No compression (passthrough)
    NONE = "vllm.v1.attention.kvcompressors.none.NoCompressor"
    # Placeholder for third-party/custom compressors - must be registered before use
    # Set to None to avoid alias with other compressor whose value is an empty string
    CUSTOM = None

    def get_path(self, include_classname: bool = True) -> str:
        """Get the class path for this compressor (respects overrides).

        Returns:
            The fully qualified class path string

        Raises:
            ValueError: If Compressor.CUSTOM is used without being registered
        """
        path = _COMPRESSOR_OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"Compressor {self.name} must be registered before use. "
                f"Use register_compressor(KVCompressorEnum.{self.name}, "
                "'your.module.YourClass')"
            )
        if not include_classname:
            path = path.rsplit(".", 1)[0]
        return path

    def get_class(self) -> "type[KVCompressorBackend]":
        """Get the compressor class (respects overrides).

        Returns:
            The compressor class

        Raises:
            ImportError: If the compressor class cannot be imported
            ValueError: If Compressor.CUSTOM is used without being registered
        """
        return resolve_obj_by_qualname(self.get_path())

    def is_overridden(self) -> bool:
        """Check if this compressor has been overridden.

        Returns:
            True if the compressor has a registered override
        """
        return self in _COMPRESSOR_OVERRIDES

    def clear_override(self) -> None:
        """Clear any override for this compressor, reverting to the default."""
        _COMPRESSOR_OVERRIDES.pop(self, None)


# Global override dictionary for runtime registration
_COMPRESSOR_OVERRIDES: dict[KVCompressorEnum, str] = {}


def register_compressor(
    compressor: KVCompressorEnum,
    class_path: str | None = None,
) -> Callable[[type], type]:
    """Register or override a KV compressor implementation.

    Args:
        compressor: The KVCompressorEnum member to register
        class_path: Optional class path. If not provided and used as
            decorator, will be auto-generated from the class.

    Returns:
        Decorator function if class_path is None, otherwise a no-op

    Examples:
        # Override an existing compressor
        @register_compressor(KVCompressorEnum.NONE)
        class MyCustomNoCompressor:
            ...

        # Register a custom third-party compressor
        @register_compressor(KVCompressorEnum.CUSTOM)
        class MyCustomCompressor:
            ...

        # Direct registration
        register_compressor(
            KVCompressorEnum.CUSTOM,
            "my.module.MyCustomCompressor"
        )
    """

    def decorator(cls: type) -> type:
        _COMPRESSOR_OVERRIDES[compressor] = f"{cls.__module__}.{cls.__qualname__}"
        return cls

    if class_path is not None:
        _COMPRESSOR_OVERRIDES[compressor] = class_path
        return lambda x: x

    return decorator


# =============================================================================
# Abstract Base Class
# =============================================================================


class KVCompressorBackend(ABC):
    """Abstract base class for KV cache compressors.

    A KV compressor manages per-request state and can compress/modify
    the KV cache as requests are processed. Implementations must handle:
    - Adding new requests
    - Removing completed requests
    - Processing sequence increments (e.g., after each token generation)
    """

    # Class-level configuration
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Return the name of this compressor backend.

        Returns:
            A string identifier for this compressor.
        """
        raise NotImplementedError

    @abstractmethod
    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        """Initialize the KV compressor.

        Args:
            vllm_config: The vLLM configuration object.
            device: The device to use for tensor operations.
        """
        raise NotImplementedError
    
    def evict_block_id(
        self, 
        block_id: int
    ):
        """
        Docstring for evict_block_id
        
        :param self: Description
        :param block_id: Description
        :type block_id: int

        Manually evict a block ID
        """
        pass

    # -------------------------------------------------------------------------
    # Request Lifecycle Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def add_request(
        self,
        request_id: str,
        seq_len: int,
    ) -> None:
        """Called when a new request is added to the scheduler.

        Implementations should initialize any per-request state needed
        for compression tracking.

        Args:
            request_id: Unique identifier for the request.
            seq_len: Initial sequence length of the request (prompt length).
        """
        raise NotImplementedError

    @abstractmethod
    def remove_request(
        self,
        request_id: str,
    ) -> None:
        """Called when a request is finished and removed from the scheduler.

        Implementations should clean up any per-request state.

        Args:
            request_id: Unique identifier for the request to remove.
        """
        raise NotImplementedError

    @abstractmethod
    def on_seq_increment(
        self,
        request_id: str,
        new_seq_len: int,
    ) -> None:
        """Called when a request's sequence length is incremented by 1.

        This is typically called after each token generation step.
        Implementations can use this to update compression state,
        trigger compression operations, etc.

        Args:
            request_id: Unique identifier for the request.
            new_seq_len: The new sequence length after the increment.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Optional Batch-Level Methods
    # -------------------------------------------------------------------------

    def on_batch_start(
        self,
        request_ids: list[str],
    ) -> None:
        """Called at the start of processing a batch.

        Optional hook for batch-level initialization.

        Args:
            request_ids: List of request IDs in the current batch.
        """
        pass

    def on_batch_end(
        self,
        request_ids: list[str],
    ) -> None:
        """Called at the end of processing a batch.

        Optional hook for batch-level finalization or compression triggers.

        Args:
            request_ids: List of request IDs in the current batch.
        """
        pass

    # -------------------------------------------------------------------------
    # Validation and Utility Methods
    # -------------------------------------------------------------------------

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        """Check if this compressor supports the given dtype.

        Args:
            dtype: The torch dtype to check.

        Returns:
            True if the dtype is supported.
        """
        return dtype in cls.supported_dtypes

    @classmethod
    def validate_configuration(
        cls,
        dtype: torch.dtype,
    ) -> list[str]:
        """Validate that the configuration is supported by this compressor.

        Args:
            dtype: The model dtype.

        Returns:
            A list of reasons why the configuration is invalid.
            Empty list if valid.
        """
        invalid_reasons = []
        if not cls.supports_dtype(dtype):
            invalid_reasons.append(f"dtype {dtype} not supported")
        return invalid_reasons
