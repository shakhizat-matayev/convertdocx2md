from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="HealthResponse")


@_attrs_define
class HealthResponse:
    """
    Attributes:
        status (str):
        graphrag_root (str):
        output_dir (str):
        timeout_seconds (float):
        max_concurrent_requests (int):
        max_query_chars (int):
        artifacts_container (str):
        artifacts_prefix (str):
        artifact_cache_ttl_seconds (int):
        artifacts_loaded (bool):
        artifact_cache_age_seconds (int | None | Unset):
    """

    status: str
    graphrag_root: str
    output_dir: str
    timeout_seconds: float
    max_concurrent_requests: int
    max_query_chars: int
    artifacts_container: str
    artifacts_prefix: str
    artifact_cache_ttl_seconds: int
    artifacts_loaded: bool
    artifact_cache_age_seconds: int | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        status = self.status

        graphrag_root = self.graphrag_root

        output_dir = self.output_dir

        timeout_seconds = self.timeout_seconds

        max_concurrent_requests = self.max_concurrent_requests

        max_query_chars = self.max_query_chars

        artifacts_container = self.artifacts_container

        artifacts_prefix = self.artifacts_prefix

        artifact_cache_ttl_seconds = self.artifact_cache_ttl_seconds

        artifacts_loaded = self.artifacts_loaded

        artifact_cache_age_seconds: int | None | Unset
        if isinstance(self.artifact_cache_age_seconds, Unset):
            artifact_cache_age_seconds = UNSET
        else:
            artifact_cache_age_seconds = self.artifact_cache_age_seconds

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "status": status,
                "graphrag_root": graphrag_root,
                "output_dir": output_dir,
                "timeout_seconds": timeout_seconds,
                "max_concurrent_requests": max_concurrent_requests,
                "max_query_chars": max_query_chars,
                "artifacts_container": artifacts_container,
                "artifacts_prefix": artifacts_prefix,
                "artifact_cache_ttl_seconds": artifact_cache_ttl_seconds,
                "artifacts_loaded": artifacts_loaded,
            }
        )
        if artifact_cache_age_seconds is not UNSET:
            field_dict["artifact_cache_age_seconds"] = artifact_cache_age_seconds

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        status = d.pop("status")

        graphrag_root = d.pop("graphrag_root")

        output_dir = d.pop("output_dir")

        timeout_seconds = d.pop("timeout_seconds")

        max_concurrent_requests = d.pop("max_concurrent_requests")

        max_query_chars = d.pop("max_query_chars")

        artifacts_container = d.pop("artifacts_container")

        artifacts_prefix = d.pop("artifacts_prefix")

        artifact_cache_ttl_seconds = d.pop("artifact_cache_ttl_seconds")

        artifacts_loaded = d.pop("artifacts_loaded")

        def _parse_artifact_cache_age_seconds(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        artifact_cache_age_seconds = _parse_artifact_cache_age_seconds(
            d.pop("artifact_cache_age_seconds", UNSET)
        )

        health_response = cls(
            status=status,
            graphrag_root=graphrag_root,
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
            max_concurrent_requests=max_concurrent_requests,
            max_query_chars=max_query_chars,
            artifacts_container=artifacts_container,
            artifacts_prefix=artifacts_prefix,
            artifact_cache_ttl_seconds=artifact_cache_ttl_seconds,
            artifacts_loaded=artifacts_loaded,
            artifact_cache_age_seconds=artifact_cache_age_seconds,
        )

        return health_response
