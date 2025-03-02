from __future__ import annotations

from enum import Enum


class StanErrorType(Enum):
    NO_ERROR = 0
    SYNTAX_ERROR = 1
    COMPILE_ERROR = 2
    DATA_ERROR = 3
    RUNTIME_ERROR = 4

    def pretty_label(self):
        if self == StanErrorType.NO_ERROR:
            return "No error"
        elif self == StanErrorType.SYNTAX_ERROR:
            return "Syntax error"
        elif self == StanErrorType.COMPILE_ERROR:
            return "Compile error"
        elif self == StanErrorType.DATA_ERROR:
            return "Data error"
        elif self == StanErrorType.RUNTIME_ERROR:
            return "Runtime error"
        else:
            raise ValueError(f"Unknown StanErrorType: {self}")


class MessageType(Enum):
    StandardOutput = 1
    StandardError = 2
    ExceptionText = 3


class StanResultEngine(Enum):
    NONE = 0
    LAPLACE = 1
    VB = 2
    MCMC = 3
    PATHFINDER = 4

    @staticmethod
    def FromStr(value: str) -> StanResultEngine:
        if value == "laplace":
            return StanResultEngine.LAPLACE
        elif value == "vb":
            return StanResultEngine.VB
        elif value == "mcmc":
            return StanResultEngine.MCMC
        elif value == "pathfinder":
            return StanResultEngine.PATHFINDER
        else:
            raise ValueError(f"Unknown StanResultEngine: {value}")

    def txt_value(self):
        if self == StanResultEngine.LAPLACE:
            return "laplace"
        elif self == StanResultEngine.VB:
            return "vb"
        elif self == StanResultEngine.MCMC:
            return "mcmc"
        elif self == StanResultEngine.PATHFINDER:
            return "pathfinder"
        else:
            raise ValueError(f"Unknown StanResultEngine: {self}")

    def __repr__(self):
        if self == StanResultEngine.LAPLACE:
            return "Laplace approximation"
        elif self == StanResultEngine.VB:
            return "Variational Bayes"
        elif self == StanResultEngine.MCMC:
            return "MCMC"
        elif self == StanResultEngine.PATHFINDER:
            return "Pathfinder"
        else:
            raise ValueError(f"Unknown StanResultEngine: {self}")


class StanOutputScope(Enum):
    MainEffects = 1
    Covariances = 2
    FullSamples = 3
    RawOutput = 4

    def __str__(self):
        if self == StanOutputScope.MainEffects:
            return "main_effects"
        elif self == StanOutputScope.Covariances:
            return "covariances"
        elif self == StanOutputScope.FullSamples:
            return "draws"
        elif self == StanOutputScope.RawOutput:
            return "raw"
        else:
            raise ValueError(f"Unknown StanOutputScope: {self}")

    @staticmethod
    def FromStr(value: str) -> StanOutputScope:
        if value == "main_effects":
            return StanOutputScope.MainEffects
        elif value == "covariances":
            return StanOutputScope.Covariances
        elif value == "draws":
            return StanOutputScope.FullSamples
        elif value == "raw":
            return StanOutputScope.RawOutput
        else:
            raise ValueError(f"Unknown StanOutputScope: {value}")

    def txt_value(self):
        if self == StanOutputScope.MainEffects:
            return "main_effects"
        elif self == StanOutputScope.Covariances:
            return "covariances"
        elif self == StanOutputScope.FullSamples:
            return "draws"
        elif self == StanOutputScope.RawOutput:
            return "raw"
        else:
            raise ValueError(f"Unknown StanOutputScope: {self}")

    def __gt__(self, other: StanOutputScope):
        return self.value > other.value


#
# class IAbstractInferenceResult(ABC, BaseModel):
#     """
#     Class that deals mostly with the error handling but also holds the runtime of the inference.
#     """
#
#     runtime: timedelta
#     messages: dict[MessageType, str]
#     error_type: StanErrorType
#     worker_tag: str
#     data_hash: EntityHash
#     model_hash: EntityHash
#     sampling_options_hash: EntityHash
#     inference_engine: StanResultEngine
#
#     def is_error(self) -> bool:
#         return self.error_type != StanErrorType.NO_ERROR
#
#     @property
#     def error_message(self) -> str:
#         if self.error_type == StanErrorType.NO_ERROR:
#             return ""
#         ans = ""
#         if MessageType.ExceptionText in self.messages:
#             ans += self.messages[MessageType.ExceptionText]
#         if MessageType.StandardError in self.messages:
#             ans += self.messages[MessageType.StandardError]
#         return ans
#
#     @property
#     def information_message(self) -> str:
#         if MessageType.StandardOutput in self.messages:
#             return self.messages[MessageType.StandardOutput]
#         return ""
#
#     @property
#     def formatted_runtime(self) -> str:
#         return humanize.precisedelta(self.runtime)
#
#     def __repr__(self) -> str:
#         ans = (
#             f"{self.inference_engine.txt_value()} running for {self.formatted_runtime}"
#         )
#         if self.is_error():
#             ans += f" returned {self.pretty_label}:\n"
#             ans += self.error_message
#         return ans
#
#     @property
#     def inference_hash(self) -> EntityHash:
#         # Combines the model, data and runtime hashes in this order.
#         return EntityHash.FromBytes(
#             data=self.model_hash.as_bytes
#             + self.data_hash.as_bytes
#             + self.sampling_options_hash.as_bytes
#         )
