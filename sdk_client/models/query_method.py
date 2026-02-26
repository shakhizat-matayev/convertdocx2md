from enum import Enum


class QueryMethod(str, Enum):
    DRIFT = "drift"
    GLOBAL = "global"
    LOCAL = "local"

    def __str__(self) -> str:
        return str(self.value)
