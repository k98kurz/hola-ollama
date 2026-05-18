"""
E2E tests for calculate tool.
"""

import asyncio
import unittest
from hola.session.session import Session
from hola.tools.core import execute_tool_call, list_tools
from hola.tools.types import ToolCall, ToolCallFunction


async def test(session_name: str, cases: list[tuple]):
    session = Session(session_name, portable=True)
    for expression, expected in cases:
        call = ToolCall(
            function=ToolCallFunction(
                name="calculate",
                arguments={"expression": expression},
            ),
        )
        result = await execute_tool_call(call, session)
        assert result == expected, (
            f"Failed for {expression}: expected {expected}, got {result}"
        )
    session.delete()


class TestCalculate(unittest.TestCase):
    """E2E tests for calculate tool."""
    def test_tool_registration_and_schema(self):
        """Test that calculate tool is registered with correct schema."""
        tools = list_tools()
        tool_names = [t.name for t in tools]
        assert "calculate" in tool_names, tool_names

        calculate_tool = next((t for t in tools if t.name == "calculate"), None)
        assert calculate_tool is not None
        params = calculate_tool.parameters.get("properties", {})
        assert "expression" in params
        assert "precision" in params
        required = calculate_tool.parameters.get("required", [])
        assert "expression" in required
        assert "precision" not in required

    def test_basic_arithmetic_operations(self):
        """Test basic arithmetic operations."""
        BASIC_ARITHMETIC_TESTS: list[tuple[str, list[str]]] = [
            ("+ 1 2", ["3.0"]),
            ("- 5 3", ["2.0"]),
            ("* 3 4", ["12.0"]),
            ("/ 10 2", ["5.0"]),
            ("+ 1 2 3 4", ["10.0"]),
            ("- 10 1 2 3", ["4.0"]),
            ("* 2 3 4", ["24.0"]),
            ("/ 100 2 5", ["10.0"]),
            ("+ 1 2; * 3 4", ["3.0", "12.0"]),
            ("- 5", ["-5.0"]),
            ("+ 5", ["5.0"]),
        ]
        asyncio.run(test("test_calc_arithmetic", BASIC_ARITHMETIC_TESTS))

    def test_advanced_arithmetic_operations(self):
        """Test advanced arithmetic operations."""
        ADVANCED_ARITHMETIC_TESTS: list[tuple[str, list[str]]] = [
            ("% 10 3", ["1.0"]),
            ("^ 2 3", ["8.0"]),
            ("^ 4 0.5", ["2.0"]),
            ("! 5", ["120"]),
            ("! 0", ["1"]),
            ("! 1", ["1"]),
        ]
        asyncio.run(test("test_calc_advanced_arithmetic", ADVANCED_ARITHMETIC_TESTS))

    def test_constants(self):
        """Test mathematical constants."""
        CONSTANTS_TESTS: list[tuple[str, list[str]]] = [
            ("pi", ["3.141592653589793"]),
            ("e", ["2.718281828459045"]),
            ("phi", ["1.618033988749895"]),
            ("* 2 pi", ["6.283185307179586"]),
        ]
        asyncio.run(test("test_calc_constants", CONSTANTS_TESTS))

    def test_statistics_functions(self):
        """Test sum function with list and variadic syntax."""
        STATISTICS_TESTS: list[tuple[str, list[str]]] = [
            ("sum [1 2 3 4 5]", ["15.0"]),
            ("sum 1 2 3 4 5", ["15.0"]),
            ("sum [1]", ["1.0"]),
            ("avg [1 2 3 4 5]", ["3.0"]),
            ("avg 1 2 3 4 5", ["3.0"]),
            ("avg [1 2 3 4]", ["2.5"]),
            ("stdev [1 2 3 4 5]", ["1.5811388300841898"]),
            ("stdev 1 2 3 4 5", ["1.5811388300841898"]),
            ("var [1 2 3 4 5]", ["2.5"]),
            ("var 1 2 3 4 5", ["2.5"]),
            ("median [1 2 3]", ["2.0"]),
            ("median 1 2 3", ["2.0"]),
            ("median [1 2 3 4]", ["2.5"]),
            ("median 1 2 3 4", ["2.5"]),
            ("mode [1 1 2 2 2 3]", ["2.0"]),
            ("mode 1 1 2 2 2 3", ["2.0"]),
        ]
        asyncio.run(test("test_calc_sum", STATISTICS_TESTS))

    def test_trigonometric_functions(self):
        """Test trigonometric functions."""
        TRIGONOMETRIC_TESTS: list[tuple[str, list[str]]] = [
            ("sin 0", ["0.0"]),
            ("cos 0", ["1.0"]),
            ("tan 0", ["0.0"]),
            ("sin (radians 90)", ["1.0"]),
            ("cos (radians 180)", ["-1.0"]),
        ]
        asyncio.run(test("test_calc_trig", TRIGONOMETRIC_TESTS))

    def test_inverse_trigonometric_functions(self):
        """Test inverse trigonometric functions."""
        INVERSE_TRIGONOMETRIC_TESTS: list[tuple[str, list[str]]] = [
            ("asin 0", ["0.0"]),
            ("acos 1", ["0.0"]),
            ("atan 0", ["0.0"]),
        ]
        asyncio.run(test("test_calc_inverse_trig", INVERSE_TRIGONOMETRIC_TESTS))

    def test_hyperbolic_functions(self):
        """Test hyperbolic functions."""
        HYPERBOLIC_TESTS: list[tuple[str, list[str]]] = [
            ("sinh 0", ["0.0"]),
            ("cosh 0", ["1.0"]),
            ("tanh 0", ["0.0"]),
        ]
        asyncio.run(test("test_calc_hyperbolic", HYPERBOLIC_TESTS))

    def test_conversion_functions(self):
        """Test degrees and radians conversion functions."""
        CONVERSION_TESTS: list[tuple[str, list[str]]] = [
            ("radians 0", ["0.0"]),
            ("radians 180", ["3.141592653589793"]),
            ("degrees 0", ["0.0"]),
            ("degrees pi", ["180.0"]),
            ("degrees (/ pi 2)", ["90.0"]),
        ]
        asyncio.run(test("test_calc_conversion", CONVERSION_TESTS))

    def test_exponential_functions(self):
        """Test exponential and logarithmic functions."""
        EXPONENTIAL_TESTS: list[tuple[str, list[str]]] = [
            ("exp 0", ["1.0"]),
            ("exp 1", ["2.718281828459045"]),
            ("ln 1", ["0.0"]),
            ("ln e", ["1.0"]),
            ("log 100", ["4.605170185988092"]),
            ("log 100 10", ["2.0"]),
        ]
        asyncio.run(test("test_calc_exp", EXPONENTIAL_TESTS))

    def test_math_utility_functions(self):
        """Test math utility functions."""
        MATH_UTILITY_TESTS: list[tuple[str, list[str]]] = [
            ("sqrt 4", ["2.0"]),
            ("sqrt 2", ["1.4142135623730951"]),
            ("abs -5", ["5.0"]),
            ("abs 5", ["5.0"]),
            ("round 3.7", ["4.0"]),
            ("round 3.14159 2", ["3.14"]),
            ("floor 3.7", ["3"]),
            ("ceil 3.2", ["4"]),
        ]
        asyncio.run(test("test_calc_utility", MATH_UTILITY_TESTS))

    def test_gcd_function(self):
        """Test gcd function."""
        GCD_TESTS: list[tuple[str, list[str]]] = [
            ("gcd 12 8", ["4"]),
            ("gcd 12 8 4", ["4"]),
            ("gcd 7 5", ["1"]),
        ]
        asyncio.run(test("test_calc_gcd", GCD_TESTS))

    def test_lcm_function(self):
        """Test lcm function."""
        LCM_TESTS: list[tuple[str, list[str]]] = [
            ("lcm 4 6", ["12"]),
            ("lcm 4 6 8", ["24"]),
            ("lcm 5 7", ["35"]),
        ]
        asyncio.run(test("test_calc_lcm", LCM_TESTS))

    def test_variable_operations(self):
        """Test variable assignment and usage."""
        VARIABLE_TESTS: list[tuple[str, list[str]]] = [
            (":= x 5", []),
            (":= x 5; + x 3", ["8.0"]),
            (":= x 5; := y 10; + x y", ["15.0"]),
            (":= x [1 2 3]; sum x", ["6.0"]),
        ]
        asyncio.run(test("test_calc_variables", VARIABLE_TESTS))

    def test_precision_modes(self):
        """Test float and Decimal precision modes."""
        PRECISION_TESTS: list[tuple[str, int, list[str]]] = [
            ("+ 0.1 0.2", 0, ["0.30000000000000004"]),
            ("+ 0.1 0.2", 28, ["0.3"]),
            ("+ 0.1 0.2", 10, ["0.3"]),
            ("sin (radians 90)", 28, ["1"]),
            ("* pi 2", 10, ["6.283185307"]),
        ]
        async def test():
            session = Session("test_calc_precision", portable=True)
            for expression, precision, expected in PRECISION_TESTS:
                call = ToolCall(
                    function=ToolCallFunction(
                        name="calculate",
                        arguments={"expression": expression, "precision": precision},
                    ),
                )
                result = await execute_tool_call(call, session)
                assert result == expected, (
                    f"Failed for {expression} with precision={precision}: "
                    f"expected {expected}, got {result}"
                )
            session.delete()
        asyncio.run(test())

    def test_error_cases(self):
        """Test error handling."""
        ERROR_TESTS: list[tuple[str, str]] = [
            ("/ 10 0", "Division by zero"),
            ("% 10 0", "Modulo by zero"),
            ("+ x 3", "Undefined variable"),
            (":= pi 5", "Reserved identifier cannot be used as variable"),
            ("! -1", "Factorial requires non-negative integer"),
            ("! 3.5", "Factorial requires non-negative integer"),
            ("! 200", "Factorial too large"),
            ("asin 2", "asin out of domain"),
            ("acos 2", "acos out of domain"),
            ("sqrt -1", "sqrt of negative number"),
            ("ln -1", "ln of non-positive number"),
            ("log -1", "log of non-positive number"),
            ("log 100 1", "Invalid log base"),
            ("log 100 -2", "Invalid log base"),
        ]
        async def test():
            session = Session("test_calc_errors", portable=True)
            from hola.ollama.exceptions import ToolExecutionError
            for expression, expected_error in ERROR_TESTS:
                call = ToolCall(
                    function=ToolCallFunction(
                        name="calculate",
                        arguments={"expression": expression},
                    ),
                )
                with self.assertRaises(ToolExecutionError) as e:
                    await execute_tool_call(call, session)
                assert expected_error in str(e.exception), (
                    f"Expected '{expected_error}' in error for {expression}, "
                    f"got {e.exception}"
                )
            session.delete()
        asyncio.run(test())

    def test_complex_syntax_e2e(self):
        """Test complex syntax: multiple statements, nesting, sequential returns."""
        COMPLEX_SYNTAX_TESTS: list[tuple[str, list[str]]] = [
            (":= a 1; := b 2; := r1 (+ b (* a 3)); := r2 (+ b (* a 4)); r1; r2; sum r1 r2",
             ["5.0", "6.0", "11.0"]),
            ("* 2 pi", ["6.283185307179586"]),
            ("sin (/ (* pi 2) 3)", ["0.8660254037844387"]),
            (":= a 5; a; * a 2; sqrt a; sqrt (* a 4)",
             ["5.0", "10.0", "2.23606797749979", "4.47213595499958"]),
            (":= x 3; := y 4; + (* x x) (* y y)", ["25.0"]),
            (":= base 10; := height 5; := area (* base height); area; / area 2",
             ["50.0", "25.0"]),
            ("+ 1 2; * 3 4; ^ 2 3; % 10 3", ["3.0", "12.0", "8.0", "1.0"]),
        ]
        asyncio.run(test("test_calc_complex_syntax", COMPLEX_SYNTAX_TESTS))


if __name__ == "__main__":
    unittest.main()
