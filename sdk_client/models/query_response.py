from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

from ..models.query_method import QueryMethod

T = TypeVar("T", bound="QueryResponse")


@_attrs_define
class QueryResponse:
    """
    Attributes:
        request_id (str):
        method (QueryMethod):
        answer (str):
        latency_ms (int):
    """

    request_id: str
    method: QueryMethod
    answer: str
    latency_ms: int

    def to_dict(self) -> dict[str, Any]:
        request_id = self.request_id

        method = self.method.value

        answer = self.answer

        latency_ms = self.latency_ms

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "request_id": request_id,
                "method": method,
                "answer": answer,
                "latency_ms": latency_ms,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        request_id = d.pop("request_id")

        method = QueryMethod(d.pop("method"))

        answer = d.pop("answer")

        latency_ms = d.pop("latency_ms")

        query_response = cls(
            request_id=request_id,
            method=method,
            answer=answer,
            latency_ms=latency_ms,
        )

        return query_response
