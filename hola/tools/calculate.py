"""
Sandboxed mathematical expression evaluator using Polish notation.
"""

from __future__ import annotations
from decimal import Decimal, InvalidOperation, getcontext, localcontext
from typing import Any, Callable
from hola.tools.core import tool
from hola.ollama.exceptions import ToolExecutionError
import math
import re


FUNCTIONS = {
    "+", "-", "*", "/", "%", "^", "!",
    "sum", "avg", "stdev", "var", "median", "mode",
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "degrees", "radians",
    "sqrt", "abs", "round", "floor", "ceil",
    "exp", "ln", "log", "gcd", "lcm", "mod",
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

        if next_tok in (None, ";", ")", "]", ","):
            return VariableNode(token)

        if token in FUNCTIONS:
            args = []
            while self.tokenizer.has_more():
                peek = self.tokenizer.peek()
                if peek in (None, ";", ")", "]", ","):
                    break
                args.append(self._parse_expression())
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
                result = result + arg
            return result

        def sub(*args):
            if len(args) == 1:
                return -args[0]
            result = args[0]
            for arg in args[1:]:
                result = result - arg
            return result

        def mul(*args):
            result = args[0]
            for arg in args[1:]:
                result = result * arg
            return result

        def div(*args):
            if len(args) < 2:
                raise ToolExecutionError("Division requires at least 2 arguments")
            result = args[0]
            for arg in args[1:]:
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
            from collections import Counter
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
            import math as math_mod
            result = abs(int(args[0]))
            for arg in args[1:]:
                result = math_mod.gcd(result, abs(int(arg)))
            return result

        def lcm(*args):
            import math as math_mod
            result = abs(int(args[0]))
            for arg in args[1:]:
                result = result * abs(int(arg)) // math_mod.gcd(
                    result, abs(int(arg))
                )
            return result

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
        })

        return builtins

    def _is_zero(self, value: float | Decimal) -> bool:
        """Check if value is zero."""
        if isinstance(value, Decimal):
            return value == Decimal(0)
        return abs(value) < 1e-10

    def _is_integer(self, value: float | Decimal) -> bool:
        """Check if value is an integer."""
        if isinstance(value, Decimal):
            return value == value.to_integral_value()
        return abs(value - round(value)) < 1e-10

    def _convert_to_output(self, value: float | Decimal) -> str:
        """Convert value to string for output."""
        if isinstance(value, Decimal):
            result = str(value)
            if '.' in result:
                result = result.rstrip('0').rstrip('.')
            return result
        if isinstance(value, float) and value.is_integer():
            return str(value)
        return str(value)

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
                results.append(self._convert_to_output(result))

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
