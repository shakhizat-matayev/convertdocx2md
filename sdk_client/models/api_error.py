from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, BinaryIO, TextIO, TYPE_CHECKING, Generator

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from ..types import UNSET, Unset
from typing import cast






T = TypeVar("T", bound="ApiError")



@_attrs_define
class ApiError:
    """ 
        Attributes:
            error (str):
            request_id (None | str | Unset):
     """

    error: str
    request_id: None | str | Unset = UNSET





    def to_dict(self) -> dict[str, Any]:
        error = self.error

        request_id: None | str | Unset
        if isinstance(self.request_id, Unset):
            request_id = UNSET
        else:
            request_id = self.request_id


        field_dict: dict[str, Any] = {}

        field_dict.update({
            "error": error,
        })
        if request_id is not UNSET:
            field_dict["request_id"] = request_id

        return field_dict



    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        error = d.pop("error")

        def _parse_request_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        request_id = _parse_request_id(d.pop("request_id", UNSET))


        api_error = cls(
            error=error,
            request_id=request_id,
        )

        return api_error

