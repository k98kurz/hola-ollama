from asyncio import iscoroutine, sleep
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from logging import Logger
from typing import Any, Callable, get_args, get_origin, get_type_hints, Union


def type_assert(condition: bool, message: str = 'invalid type') -> None:
    if not condition:
        raise TypeError(message)


def value_assert(condition: bool, message: str = 'invalid value') -> None:
    if not condition:
        raise ValueError(message)


def _recast_numeric(value: Any, expected_type: Any) -> float | None:
    """Return `float(value)` when `value` is a non-bool `int` and
        `expected_type` is `float`, per the PEP 484 numeric tower;
        otherwise return `None`.
    """
    if  (   expected_type is float
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
        return float(value)
    return None


def _resolve_custom_post_init(cls) -> Callable | None:
    """Find the nearest user-defined `__post_init__` in `cls.__mro__`,
        unwrapping `validated_dataclass` wrappers to the custom
        `__post_init__` they captured.
    """
    for klass in cls.__mro__:
        post_init = klass.__dict__.get('__post_init__')
        if post_init is None:
            continue
        if getattr(post_init, '__is_validation_wrapper__', False):
            return getattr(post_init, '__wrapped_post_init__', None)
        return post_init
    return None


def validated_dataclass(cls):
    """Decorator that adds type validation to dataclasses.
        Validates all fields including nested objects. Applies the PEP
        484 numeric tower, recasting non-bool `int` values to `float`
        for `float`-typed fields, including elements of `list` fields
        and values of `dict` fields (normalized in place; dict keys
        are validated but not recast). Rejects `bool` values for
        `int` and `float` fields. Provides detailed error messages.
        Chains to a custom `__post_init__` (this class's own, or the
        nearest one in the MRO) after validation.
    """
    def validate_field(
            instance: Any,
            field_name: str,
            field_value: Any,
            field_type: type,
            class_name: str
        ) -> Any:
        """Validate a single field value against its type. Returns the
            normalized value, applying the PEP 484 numeric tower
            (`int` recast to `float` for `float` fields, including
            `list` elements and `dict` values, in place). Raises
            `TypeError` for an invalid type.
        """
        if field_type is Any:
            return field_value

        if field_type is int and isinstance(field_value, bool):
            type_assert(
                False,
                f"{class_name}.{field_name} expected int, got bool"
            )

        origin = get_origin(field_type)
        type_args = get_args(field_type)

        if origin is Union or (
                origin is not None and
                hasattr(origin, '__name__') and
                origin.__name__ == 'UnionType'
            ):
            if field_value is None and type(None) in type_args:
                return field_value
            for arg in type_args:
                if arg is type(None):
                    continue
                if arg is int and isinstance(field_value, bool):
                    continue
                arg_origin = get_origin(arg)
                if arg_origin is list:
                    if isinstance(field_value, list):
                        element_type = get_args(arg)[0] if get_args(arg) else Any
                        for i, element in enumerate(field_value):
                            normalized = validate_field(
                                instance,
                                f"{field_name}[{i}]",
                                element,
                                element_type,
                                class_name
                            )
                            if normalized is not element:
                                field_value[i] = normalized
                        return field_value
                recast = _recast_numeric(field_value, arg)
                if recast is not None:
                    return recast
                try:
                    if isinstance(field_value, arg):
                        return field_value
                except TypeError:
                    pass
            type_assert(
                False,
                f"{class_name}.{field_name} expected {field_type}, got "
                f"{type(field_value).__name__}"
            )

        if origin is list:
            type_assert(
                isinstance(field_value, list),
                f"{class_name}.{field_name} expected list, got "
                f"{type(field_value).__name__}"
            )
            if type_args:
                element_type = type_args[0]
                for i, element in enumerate(field_value):
                    normalized = validate_field(
                        instance,
                        f"{field_name}[{i}]",
                        element,
                        element_type,
                        class_name
                    )
                    if normalized is not element:
                        field_value[i] = normalized
            return field_value

        if origin is dict:
            type_assert(
                isinstance(field_value, dict),
                f"{class_name}.{field_name} expected dict, got "
                f"{type(field_value).__name__}"
            )
            if len(type_args) >= 2:
                key_type, value_type = type_args
                for key, value in field_value.items():
                    validate_field(
                        instance,
                        f"{field_name}.key",
                        key,
                        key_type,
                        class_name
                    )
                    normalized = validate_field(
                        instance,
                        f"{field_name}[{key!r}]",
                        value,
                        value_type,
                        class_name
                    )
                    if normalized is not value:
                        field_value[key] = normalized
            return field_value

        if isinstance(field_type, type) and issubclass(field_type, Enum):
            type_assert(
                isinstance(field_value, field_type),
                f"{class_name}.{field_name} expected {field_type.__name__}, "
                f"got {type(field_value).__name__}: {field_value!r}"
            )
            return field_value

        try:
            if is_dataclass(field_type):
                type_assert(
                    isinstance(field_value, field_type),
                    f"{class_name}.{field_name} expected {field_type.__name__}, "
                    f"got {type(field_value).__name__}"
                )
                return field_value
        except ImportError:
            pass

        if isinstance(field_type, type) and isinstance(field_value, field_type):
            return field_value

        recast = _recast_numeric(field_value, field_type)
        if recast is not None:
            return recast

        if isinstance(field_type, type):
            type_assert(
                isinstance(field_value, field_type),
                f"{class_name}.{field_name} expected {field_type.__name__}, got "
                f"{type(field_value).__name__}"
            )

        return field_value

    custom_post_init = _resolve_custom_post_init(cls)

    def __post_init__(self) -> None:
        """Validate all fields after initialization, then run the
            resolved custom `__post_init__` (if any).
        """
        type_hints = get_type_hints(self.__class__)

        for field in fields(self.__class__):
            field_name = field.name
            field_value = getattr(self, field_name, None)
            field_type = type_hints.get(field_name)

            if field_type is None:
                continue

            normalized = validate_field(
                self,
                field_name,
                field_value,
                field_type,
                self.__class__.__name__
            )
            if normalized is not None:
                setattr(self, field_name, normalized)

        if custom_post_init is not None:
            custom_post_init(self)

    __post_init__.__is_validation_wrapper__ = True
    __post_init__.__wrapped_post_init__ = custom_post_init

    cls.__post_init__ = __post_init__
    return dataclass(cls, slots=True)


class HolaError(Exception):
    """Base class for library errors not specific to any LLM."""


class ToolExecutionError(HolaError):
    """Raised for tool execution failures."""


class PermissionDeniedError(HolaError):
    """Raised when file operation is denied by permissions."""


class VerificationException(HolaError):
    """Raised when code violates the AST blacklist."""


def retry_with_backoff(
        max_tries: int = 3, base_delay: float = 5.0, delay_mult: float = 2.0,
        allowed_exception_classes: list[type] | None = None,
        logger: Logger | None = None
    ) -> Callable:
    """Decorator for a method that can be retried with a back-off. If
        `allowed_exception_classes` is not supplied, the decorated
        function will be retried on all exceptions. Each failed attempt
        is logged at warning level with its exception type and attempt
        number; the final failure is re-raised for the caller to handle.
    """
    def decorator(fn: Callable) -> Callable:
        async def inner(*args, **kwargs):
            delay = base_delay
            for i in range(1, max_tries+1):
                try:
                    res = fn(*args, **kwargs)
                    if iscoroutine(res):
                        return await res
                    return res
                except Exception as e:
                    if logger:
                        logger.warning(
                            '%s failed (attempt %d/%d): %s: %s',
                            getattr(fn, '__name__', str(fn)), i, max_tries,
                            type(e).__name__, e,
                        )
                    if i == max_tries:
                        raise
                    if  (   allowed_exception_classes
                            and type(e) not in allowed_exception_classes
                        ):
                        raise
                    await sleep(delay)
                    delay *= delay_mult
        return inner
    return decorator

