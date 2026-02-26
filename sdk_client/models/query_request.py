from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="QueryRequest")


@_attrs_define
class QueryRequest:
    """
    Attributes:
        question (str):
        community_level (int | Unset):  Default: 2.
        response_type (str | Unset):  Default: 'Multiple Paragraphs'.
        verbose (bool | Unset):  Default: False.
        dynamic_community_selection (bool | Unset):  Default: False.
    """

    question: str
    community_level: int | Unset = 2
    response_type: str | Unset = "Multiple Paragraphs"
    verbose: bool | Unset = False
    dynamic_community_selection: bool | Unset = False

    def to_dict(self) -> dict[str, Any]:
        question = self.question

        community_level = self.community_level

        response_type = self.response_type

        verbose = self.verbose

        dynamic_community_selection = self.dynamic_community_selection

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "question": question,
            }
        )
        if community_level is not UNSET:
            field_dict["community_level"] = community_level
        if response_type is not UNSET:
            field_dict["response_type"] = response_type
        if verbose is not UNSET:
            field_dict["verbose"] = verbose
        if dynamic_community_selection is not UNSET:
            field_dict["dynamic_community_selection"] = dynamic_community_selection

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        question = d.pop("question")

        community_level = d.pop("community_level", UNSET)

        response_type = d.pop("response_type", UNSET)

        verbose = d.pop("verbose", UNSET)

        dynamic_community_selection = d.pop("dynamic_community_selection", UNSET)

        query_request = cls(
            question=question,
            community_level=community_level,
            response_type=response_type,
            verbose=verbose,
            dynamic_community_selection=dynamic_community_selection,
        )

        return query_request
