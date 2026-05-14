from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, get_args, get_origin, get_type_hints, Union


def type_assert(condition: bool, message: str = 'invalid type') -> None:
    if not condition:
        raise TypeError(message)


def value_assert(condition: bool, message: str = 'invalid value') -> None:
    if not condition:
        raise ValueError(message)


def validated_dataclass(cls):
    """Decorator that adds type validation to dataclasses.
        Validates all fields including nested objects (Option D).
        Uses strict validation with no coercion (Option A).
        Provides detailed error messages (Option B).
    """

    def validate_field(
            instance: Any,
            field_name: str,
            field_value: Any,
            field_type: type,
            class_name: str
        ) -> None:
        """Validate a single field value against its type. Raises
            `TypeError` for an invalid type.
        """
        if field_type is Any:
            return

        origin = get_origin(field_type)
        type_args = get_args(field_type)

        if origin is Union or (
                origin is not None and
                hasattr(origin, '__name__') and
                origin.__name__ == 'UnionType'
            ):
            if field_value is None and type(None) in type_args:
                return
            for arg in type_args:
                if arg is type(None):
                    continue
                arg_origin = get_origin(arg)
                if arg_origin is list:
                    if isinstance(field_value, list):
                        element_type = get_args(arg)[0] if get_args(arg) else Any
                        for i, element in enumerate(field_value):
                            validate_field(
                                instance,
                                f"{field_name}[{i}]",
                                element,
                                element_type,
                                class_name
                            )
                        return
                try:
                    if isinstance(field_value, arg):
                        return
                except TypeError:
                    pass
            type_assert(
                False,
                f"{class_name}.{field_name} expected {field_type}, got "
                f"{type(field_value).__name__}"
            )
            return

        if origin is list:
            type_assert(
                isinstance(field_value, list),
                f"{class_name}.{field_name} expected list, got "
                f"{type(field_value).__name__}"
            )
            if type_args:
                element_type = type_args[0]
                for i, element in enumerate(field_value):
                    validate_field(
                        instance,
                        f"{field_name}[{i}]",
                        element,
                        element_type,
                        class_name
                    )
            return

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
                    validate_field(
                        instance,
                        f"{field_name}[{key!r}]",
                        value,
                        value_type,
                        class_name
                    )
            return

        if isinstance(field_type, type) and issubclass(field_type, Enum):
            type_assert(
                isinstance(field_value, field_type),
                f"{class_name}.{field_name} expected {field_type.__name__}, "
                f"got {type(field_value).__name__}: {field_value!r}"
            )
            return

        try:
            if is_dataclass(field_type):
                type_assert(
                    isinstance(field_value, field_type),
                    f"{class_name}.{field_name} expected {field_type.__name__}, "
                    f"got {type(field_value).__name__}"
                )
                return
        except ImportError:
            pass

        if isinstance(field_type, type) and isinstance(field_value, field_type):
            return

        if isinstance(field_type, type):
            type_assert(
                isinstance(field_value, field_type),
                f"{class_name}.{field_name} expected {field_type.__name__}, got "
                f"{type(field_value).__name__}"
            )

    def __post_init__(self) -> None:
        """Validate all fields after initialization."""
        type_hints = get_type_hints(self.__class__)

        for field in fields(self.__class__):
            field_name = field.name
            field_value = getattr(self, field_name, None)
            field_type = type_hints.get(field_name)

            if field_type is None:
                continue

            validate_field(
                self,
                field_name,
                field_value,
                field_type,
                self.__class__.__name__
            )

    cls.__post_init__ = __post_init__
    return dataclass(cls, slots=True)


class PermissionDeniedError(Exception):
    """Raised when file operation is denied by permissions."""


class VerificationException(Exception):
    """Raised when code violates the AST blacklist."""
