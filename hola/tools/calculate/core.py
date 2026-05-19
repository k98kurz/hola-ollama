"""
Sandboxed mathematical expression evaluator using Polish notation.
"""

from __future__ import annotations
from collections import Counter
from decimal import Decimal, InvalidOperation, getcontext, localcontext
from typing import Any, Callable
from hola.tools.core import tool
from hola.ollama.exceptions import ToolExecutionError
from .linear import (
    _use_numpy, _use_float_mode, _is_scalar, _is_vector, _is_matrix,
    Vector, Matrix,
    _norm_numpy, _dot_numpy, _cross_numpy, _unit_numpy, _angle_numpy,
    _transpose_numpy, _det_numpy, _inv_numpy, _trace_numpy, _rank_numpy,
    _solve_numpy, _lu_numpy, _qr_numpy, _matmul_numpy,
    _norm_float, _dot_float, _cross_float, _unit_float, _angle_float,
    _transpose_float, _det_float, _inv_float, _trace_float, _rank_float,
    _solve_float, _lu_float, _qr_float, _matmul_float,
    _add_vectors_float, _add_matrices_float, _add_vector_scalar_float,
    _add_matrix_scalar_float, _sub_vectors_float, _sub_matrices_float,
    _sub_vector_scalar_float, _sub_matrix_scalar_float,
    _mul_vector_scalar_float, _mul_matrix_scalar_float,
    _div_vector_scalar_float, _div_matrix_scalar_float,
    set_numpy_mode,
)
import math
import math as math_mod
import re


try:
    import numpy as np
except ImportError:
    pass


FUNCTIONS = {
    "+", "-", "*", "/", "%", "^", "!",
    "sum", "avg", "stdev", "var", "median", "mode",
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "degrees", "radians",
    "sqrt", "abs", "round", "floor", "ceil",
    "exp", "ln", "log", "gcd", "lcm", "mod",
    "norm", "dot", "cross", "unit", "angle",
    "transpose", "det", "inv", "trace", "rank",
    "solve", "lu", "qr", "matmul",
}
CONSTANTS = {"pi", "e", "phi"}
RESERVED_IDENTIFIERS = CONSTANTS.union(FUNCTIONS)


class AssignmentNode:
    """AST node for variable assignment."""
    def __init__(self, variable: str, value: Any) -> None:
        self.variable = variable
        self.value = value


class FunctionCallNode:
    """AST node for function or operator call."""
    def __init__(self, name: str, args: list[Any]) -> None:
        self.name = name
        self.args = args


class ListLiteralNode:
    """AST node for list literal."""
    def __init__(self, elements: list[Any]) -> None:
        self.elements = elements


class NumberLiteralNode:
    """AST node for number literal."""
    def __init__(self, value: float | Decimal) -> None:
        self.value = value


class VariableNode:
    """AST node for variable reference."""
    def __init__(self, name: str) -> None:
        self.name = name


class Tokenizer:
    """Tokenizer for Polish notation expressions."""
    def __init__(self, expression: str, precision: int) -> None:
        self.expression = expression
        self.precision = precision
        self.tokens: list[str] = []
        self.position = 0
        self._pre_pass()
        self._tokenize()

    def _pre_pass(self) -> None:
        """Insert whitespace around delimiters for easier parsing."""
        chars_to_surround = "()[];,"
        result = []
        for char in self.expression:
            if char in chars_to_surround:
                result.append(f" {char} ")
            else:
                result.append(char)
        self.expression = "".join(result)

    def _tokenize(self) -> None:
        """Split on whitespace and convert number tokens."""
        self.tokens = [
            token
            for token in self.expression.split()
            if token
        ]
        for i, token in enumerate(self.tokens):
            if self._is_number(token):
                if self.precision > 0:
                    try:
                        self.tokens[i] = Decimal(token)
                    except InvalidOperation as e:
                        raise ToolExecutionError(
                            f"Invalid number at position {i}: {token}"
                        ) from e
                else:
                    self.tokens[i] = float(token)

    def _is_number(self, token: str) -> bool:
        """Check if token is a numeric literal."""
        try:
            float(token)
            return True
        except ValueError:
            return False

    def peek(self) -> str | None:
        """Peek at next token without consuming."""
        if self.position >= len(self.tokens):
            return None
        token = self.tokens[self.position]
        if isinstance(token, (float, Decimal)):
            return "NUMBER"
        return token

    def peek_next(self) -> str | None:
        """Peek at token after current without consuming."""
        if self.position + 1 >= len(self.tokens):
            return None
        token = self.tokens[self.position + 1]
        if isinstance(token, (float, Decimal)):
            return "NUMBER"
        return token

    def consume(self) -> str | float | Decimal:
        """Consume and return next token."""
        if self.position >= len(self.tokens):
            raise ToolExecutionError("Unexpected end of input")
        token = self.tokens[self.position]
        self.position += 1
        return token

    def has_more(self) -> bool:
        """Check if more tokens remain."""
        return self.position < len(self.tokens)


class Parser:
    """Parser for token stream into AST nodes."""
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def parse(self) -> list[Any]:
        """Parse the token stream into a list of AST nodes."""
        nodes = []
        while self.tokenizer.has_more():
            token = self.tokenizer.peek()
            if token == ";":
                self.tokenizer.consume()
                continue
            nodes.append(self._parse_expression())
        return nodes

    def _parse_expression(self) -> Any:
        """Parse a single expression."""
        if not self.tokenizer.has_more():
            raise ToolExecutionError("Unexpected end of input")

        position = self.tokenizer.position
        if position >= len(self.tokenizer.tokens):
            raise ToolExecutionError("Unexpected end of input")

        token = self.tokenizer.tokens[position]

        if isinstance(token, (float, Decimal)):
            return self._parse_value()

        if not isinstance(token, str):
            raise ToolExecutionError(f"Unexpected token: {token}")

        if token == "(":
            return self._parse_parenthesized()
        elif token == "[":
            return self._parse_list()
        elif token == ":=":
            return self._parse_assignment()
        elif token == ":":
            next_token = None
            if self.tokenizer.position + 1 < len(self.tokenizer.tokens):
                next_token = self.tokenizer.tokens[self.tokenizer.position + 1]
            if isinstance(next_token, str) and next_token == "=":
                return self._parse_assignment()
            else:
                return self._parse_value()
        elif token == ";":
            raise ToolExecutionError("Unexpected semicolon")
        elif token in (")", "]", ","):
            raise ToolExecutionError(f"Unexpected token: {token}")
        else:
            result = self._parse_identifier()
            return result

    def _parse_identifier(self, is_toplevel: bool = False) -> Any:
        """Parse an identifier as variable or function call."""
        token = self.tokenizer.consume()

        if not isinstance(token, str):
            raise ToolExecutionError(f"Invalid identifier: {token}")

        next_tok = self.tokenizer.peek()

        if next_tok in (None, ";", ")", "]"):
            return VariableNode(token)

        if token in FUNCTIONS:
            args = []
            while self.tokenizer.has_more():
                peek = self.tokenizer.peek()
                if peek in (None, ";", ")", "]"):
                    break
                args.append(self._parse_expression())
                if self.tokenizer.peek() == ",":
                    self.tokenizer.consume()
            return FunctionCallNode(token, args)

        return VariableNode(token)

    def _starts_value(self, token: str | None) -> bool:
        """Check if token starts a value (NUMBER, IDENTIFIER, [, ()."""
        if token is None:
            return False
        if token == "NUMBER":
            return True
        if token in ("[", "("):
            return True
        if isinstance(token, str) and token not in (";", ")", "]", ","):
            return True
        return False

    def _parse_assignment(self) -> AssignmentNode:
        """Parse assignment: := variable value."""
        current_token = self.tokenizer.peek()

        if current_token == ":=":
            self.tokenizer.consume()
        elif current_token == ":":
            self.tokenizer.consume()
            if self.tokenizer.peek() != "=":
                raise ToolExecutionError("Expected := for assignment")
            self.tokenizer.consume()
        else:
            raise ToolExecutionError("Expected := for assignment")

        token = self.tokenizer.peek()
        if token is None or not isinstance(token, str):
            raise ToolExecutionError("Expected variable name after :=")

        variable = self.tokenizer.consume()
        if not isinstance(variable, str):
            raise ToolExecutionError(f"Invalid variable name: {variable}")

        if variable in RESERVED_IDENTIFIERS:
            raise ToolExecutionError(
                f"Reserved identifier cannot be used as variable: {variable}"
            )

        value = self._parse_expression()
        return AssignmentNode(variable, value)

    def _parse_value(self) -> Any:
        """Parse a value: number, variable, or function call."""
        token = self.tokenizer.consume()

        if isinstance(token, (float, Decimal)):
            return NumberLiteralNode(token)

        if not isinstance(token, str):
            raise ToolExecutionError(f"Invalid token: {token}")

        next_tok = self.tokenizer.peek()

        if next_tok in (None, ";", ")", "]", ","):
            return VariableNode(token)

        args = []
        while self.tokenizer.has_more() and self.tokenizer.peek() not in (
            ";", ")", "]", ","
        ):
            args.append(self._parse_expression())

        return FunctionCallNode(token, args)

    def _parse_parenthesized(self) -> Any:
        """Parse parenthesized expression."""
        if self.tokenizer.peek() != "(":
            raise ToolExecutionError("Expected (")
        self.tokenizer.consume()

        node = self._parse_expression()

        if self.tokenizer.peek() != ")":
            raise ToolExecutionError("Unclosed parenthesis")
        self.tokenizer.consume()

        return node

    def _parse_list(self) -> Any:
        """Parse list literal."""
        if self.tokenizer.peek() != "[":
            raise ToolExecutionError("Expected [")
        self.tokenizer.consume()

        elements: list[Any] = []
        while self.tokenizer.has_more() and self.tokenizer.peek() != "]":
            elements.append(self._parse_expression())
            if self.tokenizer.peek() == ",":
                self.tokenizer.consume()

        if not elements:
            raise ToolExecutionError("Empty list not allowed")

        if self.tokenizer.peek() != "]":
            raise ToolExecutionError("Unclosed bracket")
        self.tokenizer.consume()

        return ListLiteralNode(elements)


class Evaluator:
    """Evaluator for AST nodes."""
    def __init__(
            self,
            precision: int,
        ) -> None:
        self.precision = precision
        self.context: dict[str, float | Decimal] = {}
        self.builtins = self._create_builtins()

    def _create_builtins(self) -> dict[str, Callable[..., Any]]:
        """Create builtin function registry."""
        builtins: dict[str, Callable[..., Any]] = {
            "pi": math.pi,
            "e": math.e,
            "phi": (1 + 5**0.5) / 2, # actual calculation for phi
        }

        def add(*args):
            if len(args) == 1:
                return args[0]
            result = args[0]
            for arg in args[1:]:
                if isinstance(result, (Vector, Matrix)):
                    if isinstance(arg, (Vector, Matrix)):
                        result = result.add(arg)
                    elif _is_scalar(arg):
                        other_val = Decimal(str(arg)) if self.precision > 0 else float(arg)
                        result = result.add(other_val)
                    else:
                        raise ToolExecutionError("Cannot add matrix/vector to incompatible type")
                elif isinstance(result, list):
                    if _is_vector(result) and _is_vector(arg):
                        if _use_numpy(self.precision):
                            result = [float(x) for x in (np.array(result) + np.array(arg))]
                        elif _use_float_mode(self.precision):
                            result = _add_vectors_float(result, arg)
                        else:
                            v1_dec = Vector(result)
                            v2_dec = Vector(arg)
                            result = v1_dec.add(v2_dec)
                    elif _is_matrix(result) and _is_matrix(arg):
                        if _use_numpy(self.precision):
                            result = [[float(x) for x in row] for row in (np.array(result) + np.array(arg))]
                        elif _use_float_mode(self.precision):
                            result = _add_matrices_float(result, arg)
                        else:
                            m1_dec = Matrix(result)
                            m2_dec = Matrix(arg)
                            result = m1_dec.add(m2_dec)
                    elif _is_vector(result) and _is_scalar(arg):
                        if _use_numpy(self.precision):
                            result = [float(x) for x in (np.array(result) + float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _add_vector_scalar_float(result, float(arg))
                        else:
                            v_dec = Vector(result)
                            other_val = Decimal(str(arg))
                            result = v_dec.add(other_val)
                    elif _is_matrix(result) and _is_scalar(arg):
                        if _use_numpy(self.precision):
                            result = [[float(x) for x in row] for row in (np.array(result) + float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _add_matrix_scalar_float(result, float(arg))
                        else:
                            m_dec = Matrix(result)
                            other_val = Decimal(str(arg))
                            result = m_dec.add(other_val)
                    else:
                        result = result + arg
                else:
                    result = result + arg
            return result

        def sub(*args):
            if len(args) == 1:
                return -args[0]
            result = args[0]
            for arg in args[1:]:
                if isinstance(result, (Vector, Matrix)):
                    if isinstance(arg, (Vector, Matrix)):
                        result = result.sub(arg)
                    elif _is_scalar(arg):
                        other_val = Decimal(str(arg)) if self.precision > 0 else float(arg)
                        result = result.sub(other_val)
                    else:
                        raise ToolExecutionError("Cannot subtract incompatible type from matrix/vector")
                elif isinstance(result, list):
                    if _is_vector(result) and _is_vector(arg):
                        if _use_numpy(self.precision):
                            result = [float(x) for x in (np.array(result) - np.array(arg))]
                        elif _use_float_mode(self.precision):
                            result = _sub_vectors_float(result, arg)
                        else:
                            v1_dec = Vector(result)
                            v2_dec = Vector(arg)
                            result = v1_dec.sub(v2_dec)
                    elif _is_matrix(result) and _is_matrix(arg):
                        if _use_numpy(self.precision):
                            result = [[float(x) for x in row] for row in (np.array(result) - np.array(arg))]
                        elif _use_float_mode(self.precision):
                            result = _sub_matrices_float(result, arg)
                        else:
                            m1_dec = Matrix(result)
                            m2_dec = Matrix(arg)
                            result = m1_dec.sub(m2_dec)
                    elif _is_vector(result) and _is_scalar(arg):
                        if _use_numpy(self.precision):
                            result = [float(x) for x in (np.array(result) - float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _sub_vector_scalar_float(result, float(arg))
                        else:
                            v_dec = Vector(result)
                            other_val = Decimal(str(arg))
                            result = v_dec.sub(other_val)
                    elif _is_matrix(result) and _is_scalar(arg):
                        if _use_numpy(self.precision):
                            result = [[float(x) for x in row] for row in (np.array(result) - float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _sub_matrix_scalar_float(result, float(arg))
                        else:
                            m_dec = Matrix(result)
                            other_val = Decimal(str(arg))
                            result = m_dec.sub(other_val)
                    else:
                        result = result - arg
                else:
                    result = result - arg
            return result

        def mul(*args):
            result = args[0]
            for arg in args[1:]:
                if isinstance(result, (Vector, Matrix)):
                    if _is_scalar(arg):
                        other_val = Decimal(str(arg)) if self.precision > 0 else float(arg)
                        result = result.mul(other_val)
                    else:
                        raise ToolExecutionError("Matrix/vector multiplication requires scalar (use matmul for matrix-vector multiplication)")
                elif isinstance(result, list):
                    if _is_vector(result) and _is_scalar(arg):
                        if _use_numpy(self.precision):
                            result = [float(x) for x in (np.array(result) * float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _mul_vector_scalar_float(result, float(arg))
                        else:
                            v_dec = Vector(result)
                            other_val = Decimal(str(arg))
                            result = v_dec.mul(other_val)
                    elif _is_matrix(result) and _is_scalar(arg):
                        if _use_numpy(self.precision):
                            result = [[float(x) for x in row] for row in (np.array(result) * float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _mul_matrix_scalar_float(result, float(arg))
                        else:
                            m_dec = Matrix(result)
                            other_val = Decimal(str(arg))
                            result = m_dec.mul(other_val)
                    else:
                        result = result * arg
                else:
                    result = result * arg
            return result

        def div(*args):
            if len(args) < 2:
                raise ToolExecutionError("Division requires at least 2 arguments")
            result = args[0]
            for arg in args[1:]:
                if isinstance(result, (Vector, Matrix)):
                    if _is_scalar(arg):
                        if self._is_zero(arg):
                            raise ToolExecutionError("Division by zero")
                        other_val = Decimal(str(arg)) if self.precision > 0 else float(arg)
                        result = result.div(other_val)
                    else:
                        raise ToolExecutionError("Matrix/vector division requires scalar")
                elif isinstance(result, list):
                    if _is_vector(result) and _is_scalar(arg):
                        if self._is_zero(arg):
                            raise ToolExecutionError("Division by zero")
                        if _use_numpy(self.precision):
                            result = [float(x) for x in (np.array(result) / float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _div_vector_scalar_float(result, float(arg))
                        else:
                            v_dec = Vector(result)
                            other_val = Decimal(str(arg))
                            result = v_dec.div(other_val)
                    elif _is_matrix(result) and _is_scalar(arg):
                        if self._is_zero(arg):
                            raise ToolExecutionError("Division by zero")
                        if _use_numpy(self.precision):
                            result = [[float(x) for x in row] for row in (np.array(result) / float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _div_matrix_scalar_float(result, float(arg))
                        else:
                            m_dec = Matrix(result)
                            other_val = Decimal(str(arg))
                            result = m_dec.div(other_val)
                    else:
                        if self._is_zero(arg):
                            raise ToolExecutionError("Division by zero")
                        result = result / arg
                else:
                    if self._is_zero(arg):
                        raise ToolExecutionError("Division by zero")
                    result = result / arg
            return result

        def sub(*args):
            if len(args) == 1:
                return -args[0]
            result = args[0]
            for arg in args[1:]:
                if isinstance(result, (Vector, Matrix)):
                    if isinstance(arg, (Vector, Matrix)):
                        result = result.sub(arg)
                    elif _is_scalar(arg):
                        other_val = Decimal(str(arg)) if self.precision > 0 else float(arg)
                        result = result.sub(other_val)
                    else:
                        raise ToolExecutionError("Cannot subtract incompatible type from matrix/vector")
                elif isinstance(result, list):
                    if _is_vector(result) and _is_vector(arg):
                        if _use_numpy(self.precision):
                            result = [float(x) for x in (np.array(result) - np.array(arg))]
                        elif _use_float_mode(self.precision):
                            result = _sub_vectors_float(result, arg)
                        else:
                            v1_dec = Vector(result)
                            v2_dec = Vector(arg)
                            result = v1_dec.sub(v2_dec)
                    elif _is_matrix(result) and _is_matrix(arg):
                        if _use_numpy(self.precision):
                            result = [[float(x) for x in row] for row in (np.array(result) - np.array(arg))]
                        elif _use_float_mode(self.precision):
                            result = _sub_matrices_float(result, arg)
                        else:
                            m1_dec = Matrix(result)
                            m2_dec = Matrix(arg)
                            result = m1_dec.sub(m2_dec)
                    elif _is_vector(result) and _is_scalar(arg):
                        if _use_numpy(self.precision):
                            result = [float(x) for x in (np.array(result) - float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _sub_vector_scalar_float(result, float(arg))
                        else:
                            v_dec = Vector(result)
                            other_val = Decimal(str(arg))
                            result = v_dec.sub(other_val)
                    elif _is_matrix(result) and _is_scalar(arg):
                        if _use_numpy(self.precision):
                            result = [[float(x) for x in row] for row in (np.array(result) - float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _sub_matrix_scalar_float(result, float(arg))
                        else:
                            m_dec = Matrix(result)
                            other_val = Decimal(str(arg))
                            result = m_dec.sub(other_val)
                    else:
                        result = result - arg
                else:
                    result = result - arg
            return result

        def mul(*args):
            result = args[0]
            for arg in args[1:]:
                if isinstance(result, (Vector, Matrix)):
                    if _is_scalar(arg):
                        other_val = Decimal(str(arg)) if self.precision > 0 else float(arg)
                        result = result.mul(other_val)
                    else:
                        raise ToolExecutionError("Matrix/vector multiplication requires scalar (use matmul for matrix-vector multiplication)")
                elif isinstance(result, list):
                    if _is_vector(result) and _is_scalar(arg):
                        if _use_numpy(self.precision):
                            result = [float(x) for x in (np.array(result) * float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _mul_vector_scalar_float(result, float(arg))
                        else:
                            v_dec = Vector(result)
                            other_val = Decimal(str(arg))
                            result = v_dec.mul(other_val)
                    elif _is_matrix(result) and _is_scalar(arg):
                        if _use_numpy(self.precision):
                            result = [[float(x) for x in row] for row in (np.array(result) * float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _mul_matrix_scalar_float(result, float(arg))
                        else:
                            m_dec = Matrix(result)
                            other_val = Decimal(str(arg))
                            result = m_dec.mul(other_val)
                    else:
                        result = result * arg
                else:
                    result = result * arg
            return result

        def div(*args):
            if len(args) < 2:
                raise ToolExecutionError("Division requires at least 2 arguments")
            result = args[0]
            for arg in args[1:]:
                if isinstance(result, (Vector, Matrix)):
                    if _is_scalar(arg):
                        if self._is_zero(arg):
                            raise ToolExecutionError("Division by zero")
                        other_val = Decimal(str(arg)) if self.precision > 0 else float(arg)
                        result = result.div(other_val)
                    else:
                        raise ToolExecutionError("Matrix/vector division requires scalar")
                elif isinstance(result, list):
                    if _is_vector(result) and _is_scalar(arg):
                        if self._is_zero(arg):
                            raise ToolExecutionError("Division by zero")
                        if _use_numpy(self.precision):
                            result = [float(x) for x in (np.array(result) / float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _div_vector_scalar_float(result, float(arg))
                        else:
                            v_dec = Vector(result)
                            other_val = Decimal(str(arg))
                            result = v_dec.div(other_val)
                    elif _is_matrix(result) and _is_scalar(arg):
                        if self._is_zero(arg):
                            raise ToolExecutionError("Division by zero")
                        if _use_numpy(self.precision):
                            result = [[float(x) for x in row] for row in (np.array(result) / float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _div_matrix_scalar_float(result, float(arg))
                        else:
                            m_dec = Matrix(result)
                            other_val = Decimal(str(arg))
                            result = m_dec.div(other_val)
                    else:
                        if self._is_zero(arg):
                            raise ToolExecutionError("Division by zero")
                        result = result / arg
                else:
                    if self._is_zero(arg):
                        raise ToolExecutionError("Division by zero")
                    result = result / arg
            return result

        def sub(*args):
            if len(args) == 1:
                return -args[0]
            result = args[0]
            for arg in args[1:]:
                if isinstance(result, (Vector, Matrix)):
                    if isinstance(arg, (Vector, Matrix)):
                        result = result.sub(arg)
                    elif _is_scalar(arg):
                        other_val = Decimal(str(arg)) if self.precision > 0 else float(arg)
                        result = result.sub(other_val)
                    else:
                        raise ToolExecutionError("Cannot subtract incompatible type from matrix/vector")
                elif isinstance(result, list):
                    if _is_vector(result) and _is_vector(arg):
                        if _use_numpy(self.precision):
                            result = [float(x) for x in (np.array(result) - np.array(arg))]
                        elif _use_float_mode(self.precision):
                            result = _sub_vectors_float(result, arg)
                        else:
                            v1_dec = Vector(result)
                            v2_dec = Vector(arg)
                            result = v1_dec.sub(v2_dec)
                    elif _is_matrix(result) and _is_matrix(arg):
                        if _use_numpy(self.precision):
                            result = [[float(x) for x in row] for row in (np.array(result) - np.array(arg))]
                        elif _use_float_mode(self.precision):
                            result = _sub_matrices_float(result, arg)
                        else:
                            m1_dec = Matrix(result)
                            m2_dec = Matrix(arg)
                            result = m1_dec.sub(m2_dec)
                    elif _is_vector(result) and _is_scalar(arg):
                        if _use_numpy(self.precision):
                            result = [float(x) for x in (np.array(result) - float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _sub_vector_scalar_float(result, float(arg))
                        else:
                            v_dec = Vector(result)
                            other_val = Decimal(str(arg))
                            result = v_dec.sub(other_val)
                    elif _is_matrix(result) and _is_scalar(arg):
                        if _use_numpy(self.precision):
                            result = [[float(x) for x in row] for row in (np.array(result) - float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _sub_matrix_scalar_float(result, float(arg))
                        else:
                            m_dec = Matrix(result)
                            other_val = Decimal(str(arg))
                            result = m_dec.sub(other_val)
                    else:
                        result = result - arg
                else:
                    result = result - arg
            return result

        def mul(*args):
            result = args[0]
            for arg in args[1:]:
                if isinstance(result, (Vector, Matrix)):
                    if _is_scalar(arg):
                        other_val = Decimal(str(arg)) if self.precision > 0 else float(arg)
                        result = result.mul(other_val)
                    else:
                        raise ToolExecutionError("Matrix/vector multiplication requires scalar (use matmul for matrix-vector multiplication)")
                elif isinstance(result, list):
                    if _is_vector(result) and _is_scalar(arg):
                        if _use_numpy(self.precision):
                            result = [float(x) for x in (np.array(result) * float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _mul_vector_scalar_float(result, float(arg))
                        else:
                            v_dec = Vector(result)
                            other_val = Decimal(str(arg))
                            result = v_dec.mul(other_val)
                    elif _is_matrix(result) and _is_scalar(arg):
                        if _use_numpy(self.precision):
                            result = [[float(x) for x in row] for row in (np.array(result) * float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _mul_matrix_scalar_float(result, float(arg))
                        else:
                            m_dec = Matrix(result)
                            other_val = Decimal(str(arg))
                            result = m_dec.mul(other_val)
                    else:
                        result = result * arg
                else:
                    result = result * arg
            return result

        def div(*args):
            if len(args) < 2:
                raise ToolExecutionError("Division requires at least 2 arguments")
            result = args[0]
            for arg in args[1:]:
                if isinstance(result, (Vector, Matrix)):
                    if _is_scalar(arg):
                        if self._is_zero(arg):
                            raise ToolExecutionError("Division by zero")
                        other_val = Decimal(str(arg)) if self.precision > 0 else float(arg)
                        result = result.div(other_val)
                    else:
                        raise ToolExecutionError("Matrix/vector division requires scalar")
                elif isinstance(result, list):
                    if _is_vector(result) and _is_scalar(arg):
                        if self._is_zero(arg):
                            raise ToolExecutionError("Division by zero")
                        if _use_numpy(self.precision):
                            result = [float(x) for x in (np.array(result) / float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _div_vector_scalar_float(result, float(arg))
                        else:
                            v_dec = Vector(result)
                            other_val = Decimal(str(arg))
                            result = v_dec.div(other_val)
                    elif _is_matrix(result) and _is_scalar(arg):
                        if self._is_zero(arg):
                            raise ToolExecutionError("Division by zero")
                        if _use_numpy(self.precision):
                            result = [[float(x) for x in row] for row in (np.array(result) / float(arg))]
                        elif _use_float_mode(self.precision):
                            result = _div_matrix_scalar_float(result, float(arg))
                        else:
                            m_dec = Matrix(result)
                            other_val = Decimal(str(arg))
                            result = m_dec.div(other_val)
                    else:
                        if self._is_zero(arg):
                            raise ToolExecutionError("Division by zero")
                        result = result / arg
                else:
                    if self._is_zero(arg):
                        raise ToolExecutionError("Division by zero")
                    result = result / arg
            return result

        def mod(*args):
            if len(args) < 2:
                raise ToolExecutionError("Modulo requires at least 2 arguments")
            result = args[0]
            for arg in args[1:]:
                if self._is_zero(arg):
                    raise ToolExecutionError("Modulo by zero")
                result = result % arg
            return result

        def power(base, exp):
            return base ** exp

        def factorial(n):
            if n < 0 or not self._is_integer(n):
                raise ToolExecutionError(
                    f"Factorial requires non-negative integer, got {n}"
                )
            if n > 170:
                raise ToolExecutionError(
                    f"Factorial too large: {n}"
                )
            result = 1
            for i in range(2, int(n) + 1):
                result *= i
            return result

        def sum_impl(*args):
            if len(args) == 1 and isinstance(args[0], list):
                vals = args[0]
            else:
                vals = list(args)
            return sum(vals)

        def avg(*args):
            if len(args) == 1 and isinstance(args[0], list):
                vals = args[0]
            else:
                vals = list(args)
            if not vals:
                raise ToolExecutionError("Average of empty list")
            return sum(vals) / len(vals)

        def stdev(*args):
            if len(args) == 1 and isinstance(args[0], list):
                vals = args[0]
            else:
                vals = list(args)
            if len(vals) < 2:
                raise ToolExecutionError("Stdev requires at least 2 values")
            mean = sum(vals) / len(vals)
            variance = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
            return variance ** 0.5

        def variance(*args):
            if len(args) == 1 and isinstance(args[0], list):
                vals = args[0]
            else:
                vals = list(args)
            if len(vals) < 2:
                raise ToolExecutionError("Variance requires at least 2 values")
            mean = sum(vals) / len(vals)
            return sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)

        def median(*args):
            if len(args) == 1 and isinstance(args[0], list):
                vals = sorted(args[0])
            else:
                vals = sorted(list(args))
            if not vals:
                raise ToolExecutionError("Median of empty list")
            n = len(vals)
            if n % 2 == 1:
                return vals[n // 2]
            return (vals[n // 2 - 1] + vals[n // 2]) / 2

        def mode(*args):
            if len(args) == 1 and isinstance(args[0], list):
                vals = args[0]
            else:
                vals = list(args)
            if not vals:
                raise ToolExecutionError("Mode of empty list")
            counts = Counter(vals)
            max_count = max(counts.values())
            modes = [v for v, c in counts.items() if c == max_count]
            return modes[0]

        def sin(x):
            return math.sin(float(x))

        def cos(x):
            return math.cos(float(x))

        def tan(x):
            return math.tan(float(x))

        def asin(x):
            fx = float(x)
            if fx < -1 or fx > 1:
                raise ToolExecutionError(
                    f"asin out of domain [-1, 1]: {x}"
                )
            return math.asin(fx)

        def acos(x):
            fx = float(x)
            if fx < -1 or fx > 1:
                raise ToolExecutionError(
                    f"acos out of domain [-1, 1]: {x}"
                )
            return math.acos(fx)

        def atan(x):
            return math.atan(float(x))

        def sinh(x):
            return math.sinh(float(x))

        def cosh(x):
            return math.cosh(float(x))

        def tanh(x):
            return math.tanh(float(x))

        def degrees(x):
            """From radians to degrees."""
            return math.degrees(float(x))

        def radians(x):
            """From degrees to radians."""
            return math.radians(float(x))

        def sqrt(x):
            if x < 0:
                raise ToolExecutionError(f"sqrt of negative number: {x}")
            return x ** 0.5

        def abs_impl(x):
            return abs(x)

        def round_impl(x, digits=0):
            return round(x, int(digits))

        def floor(x):
            return math.floor(float(x))

        def ceil(x):
            return math.ceil(float(x))

        def exp(x):
            return math.exp(float(x))

        def ln(x):
            if x <= 0:
                raise ToolExecutionError(f"ln of non-positive number: {x}")
            return math.log(float(x))

        def log(x, base=None):
            if x <= 0:
                raise ToolExecutionError(f"log of non-positive number: {x}")
            if base is None or self._is_zero(base - 2.718281828459045):
                return math.log(float(x))
            if base <= 0 or base == 1:
                raise ToolExecutionError(
                    f"Invalid log base: {base}"
                )
            return math.log(float(x), float(base))

        def gcd(*args):
            result = abs(int(args[0]))
            for arg in args[1:]:
                result = math_mod.gcd(result, abs(int(arg)))
            return result

        def lcm(*args):
            result = abs(int(args[0]))
            for arg in args[1:]:
                result = result * abs(int(arg)) // math_mod.gcd(
                    result, abs(int(arg))
                )
            return result

        def norm_impl(x):
            if _use_numpy(self.precision):
                if _is_vector(x):
                    return _norm_numpy(x)
                if _is_matrix(x):
                    return _norm_numpy([r for row in x for r in row])
            if _use_float_mode(self.precision):
                if _is_vector(x):
                    return _norm_float(x)
                if _is_matrix(x):
                    return _norm_float([r for row in x for r in row])
            if isinstance(x, Vector):
                return x.norm()
            if isinstance(x, Matrix):
                return sum(x.data[i][j] ** 2 for i in range(x.rows()) for j in range(x.cols())).sqrt()
            raise ToolExecutionError("norm requires vector or matrix")

        def dot_impl(v1, v2):
            if _use_numpy(self.precision):
                return _dot_numpy(v1, v2)
            if _use_float_mode(self.precision):
                return _dot_float(v1, v2)
            if isinstance(v1, Vector) and isinstance(v2, Vector):
                return v1.dot(v2)
            if _is_vector(v1) and _is_vector(v2):
                v1_dec = Vector(v1)
                v2_dec = Vector(v2)
                return v1_dec.dot(v2_dec)
            raise ToolExecutionError("dot requires two vectors")

        def cross_impl(v1, v2):
            if _use_numpy(self.precision):
                return _cross_numpy(v1, v2)
            if _use_float_mode(self.precision):
                return _cross_float(v1, v2)
            if isinstance(v1, Vector) and isinstance(v2, Vector):
                result = v1.cross(v2)
                return result.data if _is_vector(v1) else result
            if _is_vector(v1) and _is_vector(v2):
                v1_dec = Vector(v1)
                v2_dec = Vector(v2)
                result = v1_dec.cross(v2_dec)
                return result.data
            raise ToolExecutionError("cross requires two 3D vectors")

        def unit_impl(v):
            if _use_numpy(self.precision):
                return _unit_numpy(v)
            if _use_float_mode(self.precision):
                return _unit_float(v)
            if isinstance(v, Vector):
                result = v.unit()
                return result.data if _is_vector(v) else result
            if _is_vector(v):
                v_dec = Vector(v)
                result = v_dec.unit()
                return result.data
            raise ToolExecutionError("unit requires vector")

        def angle_impl(v1, v2):
            if _use_numpy(self.precision):
                return _angle_numpy(v1, v2)
            if _use_float_mode(self.precision):
                return _angle_float(v1, v2)
            if isinstance(v1, Vector) and isinstance(v2, Vector):
                return v1.angle(v2)
            if _is_vector(v1) and _is_vector(v2):
                v1_dec = Vector(v1)
                v2_dec = Vector(v2)
                return v1_dec.angle(v2_dec)
            raise ToolExecutionError("angle requires two vectors")

        def transpose_impl(m):
            if _use_numpy(self.precision):
                return _transpose_numpy(m)
            if _use_float_mode(self.precision):
                return _transpose_float(m)
            if isinstance(m, Matrix):
                result = m.transpose()
                return result.data if _is_matrix(m) else result
            if _is_matrix(m):
                m_dec = Matrix(m)
                result = m_dec.transpose()
                return result.data
            raise ToolExecutionError("transpose requires matrix")

        def det_impl(m):
            if _use_numpy(self.precision):
                return _det_numpy(m)
            if _use_float_mode(self.precision):
                return _det_float(m)
            if isinstance(m, Matrix):
                return float(m.det())
            if _is_matrix(m):
                m_dec = Matrix(m)
                return float(m_dec.det())
            raise ToolExecutionError("det requires square matrix")

        def inv_impl(m):
            if _use_numpy(self.precision):
                return _inv_numpy(m)
            if _use_float_mode(self.precision):
                return _inv_float(m)
            if isinstance(m, Matrix):
                result = m.inv()
                return result.data if _is_matrix(m) else result
            if _is_matrix(m):
                m_dec = Matrix(m)
                result = m_dec.inv()
                return result.data
            raise ToolExecutionError("inv requires non-singular square matrix")

        def trace_impl(m):
            if _use_numpy(self.precision):
                return _trace_numpy(m)
            if _use_float_mode(self.precision):
                return _trace_float(m)
            if isinstance(m, Matrix):
                return float(m.trace())
            if _is_matrix(m):
                m_dec = Matrix(m)
                return float(m_dec.trace())
            raise ToolExecutionError("trace requires square matrix")

        def rank_impl(m):
            if _use_numpy(self.precision):
                return float(_rank_numpy(m))
            if _use_float_mode(self.precision):
                return float(_rank_float(m))
            if isinstance(m, Matrix):
                return float(m.rank())
            if _is_matrix(m):
                m_dec = Matrix(m)
                return float(m_dec.rank())
            raise ToolExecutionError("rank requires matrix")

        def solve_impl(A, b):
            if _use_numpy(self.precision):
                return _solve_numpy(A, b)
            if _use_float_mode(self.precision):
                return _solve_float(A, b)
            if isinstance(A, Matrix) and isinstance(b, Vector):
                result = A._solve_system(b)
                return result.data
            if _is_matrix(A) and _is_vector(b):
                A_dec = Matrix(A)
                b_dec = Vector(b)
                result = A_dec._solve_system(b_dec)
                return result.data
            raise ToolExecutionError("solve requires matrix and vector")

        def lu_impl(m):
            if _use_numpy(self.precision):
                result = _lu_numpy(m)
                return result
            if _use_float_mode(self.precision):
                result = _lu_float(m)
                return result
            if isinstance(m, Matrix):
                L, U = m._lu_decomposition()
                return [L.data, U.data]
            if _is_matrix(m):
                m_dec = Matrix(m)
                L, U = m_dec._lu_decomposition()
                return [L.data, U.data]
            raise ToolExecutionError("lu requires matrix")

        def qr_impl(m):
            if _use_numpy(self.precision):
                result = _qr_numpy(m)
                return result
            if _use_float_mode(self.precision):
                result = _qr_float(m)
                return result
            if isinstance(m, Matrix):
                Q, R = m._qr_decomposition()
                return [Q.data, R.data]
            if _is_matrix(m):
                m_dec = Matrix(m)
                Q, R = m_dec._qr_decomposition()
                return [Q.data, R.data]
            raise ToolExecutionError("qr requires matrix")

        def matmul_impl(A, B):
            if _use_numpy(self.precision):
                return _matmul_numpy(A, B)
            if _use_float_mode(self.precision):
                return _matmul_float(A, B)
            if isinstance(A, Matrix):
                result = A.matmul(B)
                if isinstance(result, Vector):
                    return result.data
                return result.data if _is_matrix(A) else result
            if _is_matrix(A):
                A_dec = Matrix(A)
                result = A_dec.matmul(B)
                return result.data
            raise ToolExecutionError("matmul requires matrix")

        builtins.update({
            "+": add,
            "-": sub,
            "*": mul,
            "/": div,
            "%": mod,
            "^": power,
            "!": factorial,
            "sum": sum_impl,
            "avg": avg,
            "stdev": stdev,
            "var": variance,
            "median": median,
            "mode": mode,
            "sin": sin,
            "cos": cos,
            "tan": tan,
            "asin": asin,
            "acos": acos,
            "atan": atan,
            "sinh": sinh,
            "cosh": cosh,
            "tanh": tanh,
            "degrees": degrees,
            "radians": radians,
            "sqrt": sqrt,
            "abs": abs_impl,
            "round": round_impl,
            "floor": floor,
            "ceil": ceil,
            "exp": exp,
            "ln": ln,
            "log": log,
            "gcd": gcd,
            "lcm": lcm,
            "mod": mod,
            "norm": norm_impl,
            "dot": dot_impl,
            "cross": cross_impl,
            "unit": unit_impl,
            "angle": angle_impl,
            "transpose": transpose_impl,
            "det": det_impl,
            "inv": inv_impl,
            "trace": trace_impl,
            "rank": rank_impl,
            "solve": solve_impl,
            "lu": lu_impl,
            "qr": qr_impl,
            "matmul": matmul_impl,
        })

        return builtins

    def _is_zero(self, value: float | Decimal) -> bool:
        """Check if value is zero."""
        if isinstance(value, Decimal):
            return value == Decimal(0)
        if _is_scalar(value):
            return abs(value) < 1e-10
        return False

    def _is_integer(self, value: float | Decimal) -> bool:
        """Check if value is an integer."""
        if isinstance(value, Decimal):
            return value == value.to_integral_value()
        return abs(value - round(value)) < 1e-10

    def _convert_scalar(self, value: float | Decimal) -> str:
        """Convert scalar value to string for output."""
        if isinstance(value, Decimal):
            result = str(value)
            if '.' in result:
                result = result.rstrip('0').rstrip('.')
            return result
        if isinstance(value, float) and value.is_integer():
            return str(value)
        return str(value)

    def _convert_to_output(self, value: float | Decimal | Vector | Matrix | list) -> str | list[str]:
        """Convert value to output format."""
        def _convert_vector_element(x: float | Decimal) -> int | float:
            """Convert vector element for list representation."""
            if isinstance(x, Decimal):
                return int(x) if (x % 1) == 0 else float(x)
            return x

        def _convert_matrix_element(x: float | Decimal) -> float:
            """Convert matrix element for list representation (always float)."""
            return float(x) if isinstance(x, Decimal) else x

        if isinstance(value, Vector):
            if self.precision == 0:
                return [self._convert_scalar(x) for x in value.data]
            return str([_convert_vector_element(x) for x in value.data])
        if isinstance(value, Matrix):
            return str([[_convert_matrix_element(x) for x in row] for row in value.data])
        if isinstance(value, list):
            if value and _is_vector(value):
                if self.precision == 0:
                    return [self._convert_scalar(x) for x in value]
                return str([_convert_vector_element(x) for x in value])
            if value and _is_matrix(value):
                return str([[_convert_matrix_element(x) for x in row] for row in value])
            if value and all(_is_scalar(e) for e in value):
                return [self._convert_scalar(x) for x in value]
            if value and all(_is_matrix(m) for m in value):
                return str([[[_convert_matrix_element(x) for x in row] for row in m] for m in value])
            return value
        return self._convert_scalar(value)

    def _evaluate(
            self,
            node: Any,
            context: dict[str, float | Decimal],
        ) -> float | Decimal:
        """Evaluate an AST node."""
        if isinstance(node, NumberLiteralNode):
            return node.value

        if isinstance(node, VariableNode):
            if node.name in context:
                return context[node.name]
            if node.name in self.builtins:
                builtin = self.builtins[node.name]
                if isinstance(builtin, (float, int)):
                    if self.precision > 0:
                        return Decimal(str(builtin))
                    return float(builtin)
                raise ToolExecutionError(
                    f"Cannot use builtin as value: {node.name}"
                )
            raise ToolExecutionError(
                f"Undefined variable: {node.name}"
            )

        if isinstance(node, FunctionCallNode):
            args = [
                self._evaluate(arg, context)
                for arg in node.args
            ]
            if node.name not in self.builtins:
                raise ToolExecutionError(
                    f"Unknown function: {node.name}"
                )
            func = self.builtins[node.name]
            if callable(func):
                result = func(*args)
                if self.precision > 0 and isinstance(result, float):
                    return Decimal(str(result))
                return result
            raise ToolExecutionError(
                f"Cannot call builtin constant as function: {node.name}"
            )

        if isinstance(node, ListLiteralNode):
            elements = [
                self._evaluate(elem, context)
                for elem in node.elements
            ]
            if elements and all(_is_scalar(e) for e in elements):
                if _use_numpy(self.precision):
                    return [float(e) for e in elements]
                if _use_float_mode(self.precision):
                    return [float(e) for e in elements]
                return Vector(elements)
            elif elements and all(_is_vector(e) or isinstance(e, Vector) for e in elements):
                if _use_numpy(self.precision):
                    return [[float(x) for x in row] for row in elements]
                if _use_float_mode(self.precision):
                    return [[float(x) for x in row] if isinstance(row, list) else row for row in elements]
                return Matrix([list(row.data) if isinstance(row, Vector) else row for row in elements])
            return elements

        if isinstance(node, AssignmentNode):
            value = self._evaluate(node.value, context)
            context[node.variable] = value
            return value

        raise ToolExecutionError(
            f"Unknown node type: {type(node)}"
        )

    def evaluate(self, nodes: list[Any]) -> list[str]:
        """Evaluate list of AST nodes, returning results."""
        if self.precision > 0:
            with localcontext() as ctx:
                ctx.prec = self.precision
                return self._evaluate_nodes(nodes)

        return self._evaluate_nodes(nodes)

    def _evaluate_nodes(self, nodes: list[Any]) -> list[str]:
        """Evaluate nodes with the current Decimal context."""
        context: dict[str, float | Decimal] = {}
        results: list[str] = []

        for node in nodes:
            if isinstance(node, AssignmentNode):
                self._evaluate(node, context)
            else:
                result = self._evaluate(node, context)
                output = self._convert_to_output(result)
                if isinstance(output, list):
                    results.extend(output)
                else:
                    results.append(output)

        return results


@tool(
    name="calculate",
    description="Evaluate mathematical expressions using Polish notation. "
    "Supports variable assignment with :=, arithmetic operators (+, -, *, /, %, ^, "
    "!), statistical functions (sum, avg, stdev, var, median, mode), "
    "trigonometric functions (sin, cos, tan, asin, acos, atan, sinh, cosh, tanh), "
    "and math functions (sqrt, abs, round, floor, ceil, exp, ln, log, gcd, lcm). "
    "Variables are scoped to the tool call. Use precision>0 for Decimal arithmetic."
)
def calculate(expression: str, precision: int = 0) -> list[str]:
    """Evaluate Polish notation expression. Returns list of string results."""
    if not isinstance(expression, str):
        raise ToolExecutionError("Expression must be a string")

    if not expression.strip():
        return []

    if not isinstance(precision, int) or precision < 0:
        raise ToolExecutionError("Precision must be a non-negative integer")

    try:
        tokenizer = Tokenizer(expression, precision)
        parser = Parser(tokenizer)
        nodes = parser.parse()
        evaluator = Evaluator(precision)
        return evaluator.evaluate(nodes)
    except ToolExecutionError:
        raise
    except Exception as e:
        raise ToolExecutionError(f"Calculation error: {e}") from e
