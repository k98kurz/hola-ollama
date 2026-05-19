"""
Linear algebra support for the calculate tool.
Supports vectors, matrices, and linear algebra operations with both
Decimal (precision mode) and NumPy (float mode) implementations.
"""

from __future__ import annotations
from decimal import Decimal
from typing import Any
from hola.ollama.exceptions import ToolExecutionError
import math

_numpy_mode: bool | None = None


def set_numpy_mode(force: bool | None) -> None:
    """Set numpy mode for testing. None=auto, True=force numpy, False=force custom."""
    global _numpy_mode
    _numpy_mode = force


def _use_numpy(precision: int) -> bool:
    """Check if numpy should be used based on precision and mode setting."""
    try:
        import numpy as np
        return precision == 0 and _numpy_mode is not False
    except ImportError:
        return False


def _is_scalar(x: Any) -> bool:
    """Check if x is a scalar value (int, float, or Decimal)."""
    return isinstance(x, (int, float, Decimal))


def _is_vector(x: Any) -> bool:
    """Check if x is a vector (1D list of scalars)."""
    return isinstance(x, list) and x and all(_is_scalar(e) for e in x)


def _is_matrix(x: Any) -> bool:
    """Check if x is a matrix (2D list of lists of scalars)."""
    return (
        isinstance(x, list)
        and x
        and all(_is_vector(row) for row in x)
    )


def _is_zero(value: float | Decimal) -> bool:
    """Check if value is zero."""
    if isinstance(value, Decimal):
        return value == Decimal(0)
    return abs(value) < 1e-10


class Vector:
    """Vector class for Decimal mode linear algebra operations."""
    def __init__(self, data: list[float | Decimal]) -> None:
        if not data or not all(_is_scalar(e) for e in data):
            raise ToolExecutionError("Vector must be non-empty list of numbers")
        self.data = [Decimal(str(e)) if not isinstance(e, Decimal) else e for e in data]

    def dim(self) -> int:
        """Return vector dimension."""
        return len(self.data)

    def norm(self) -> Decimal:
        """Return Euclidean norm (L2 norm)."""
        return sum(x * x for x in self.data).sqrt()

    def dot(self, other: Vector) -> Decimal:
        """Return dot product with another vector."""
        if self.dim() != other.dim():
            raise ToolExecutionError(f"Dimension mismatch: {self.dim()} vs {other.dim()}")
        return sum(a * b for a, b in zip(self.data, other.data))

    def cross(self, other: Vector) -> Vector:
        """Return cross product (only for 3D vectors)."""
        if self.dim() != 3 or other.dim() != 3:
            raise ToolExecutionError("Cross product requires 3D vectors")
        a = self.data
        b = other.data
        return Vector([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ])

    def unit(self) -> Vector:
        """Return unit vector (normalized vector)."""
        n = self.norm()
        if _is_zero(n):
            raise ToolExecutionError("Cannot normalize zero vector")
        return Vector([x / n for x in self.data])

    def angle(self, other: Vector) -> Decimal:
        """Return angle between this and another vector (in radians)."""
        dot_val = self.dot(other)
        norms = self.norm() * other.norm()
        if _is_zero(norms):
            raise ToolExecutionError("Cannot compute angle with zero vector")
        cos_val = max(Decimal(-1), min(Decimal(1), dot_val / norms))
        return +Decimal(str(math.acos(float(cos_val))))

    def add(self, other: Vector | Decimal) -> Vector:
        """Add vector or scalar."""
        if _is_scalar(other):
            other_val = Decimal(str(other)) if not isinstance(other, Decimal) else other
            return Vector([x + other_val for x in self.data])
        if not _is_vector(other):
            raise ToolExecutionError("Vector operation requires vector or scalar")
        if self.dim() != len(other):
            raise ToolExecutionError(f"Dimension mismatch: {self.dim()} vs {len(other)}")
        other_dec = [Decimal(str(e)) if not isinstance(e, Decimal) else e for e in other]
        return Vector([a + b for a, b in zip(self.data, other_dec)])

    def sub(self, other: Vector | Decimal) -> Vector:
        """Subtract vector or scalar."""
        if _is_scalar(other):
            other_val = Decimal(str(other)) if not isinstance(other, Decimal) else other
            return Vector([x - other_val for x in self.data])
        if not _is_vector(other):
            raise ToolExecutionError("Vector operation requires vector or scalar")
        if self.dim() != len(other):
            raise ToolExecutionError(f"Dimension mismatch: {self.dim()} vs {len(other)}")
        other_dec = [Decimal(str(e)) if not isinstance(e, Decimal) else e for e in other]
        return Vector([a - b for a, b in zip(self.data, other_dec)])

    def mul(self, other: Decimal) -> Vector:
        """Multiply by scalar."""
        if not _is_scalar(other):
            raise ToolExecutionError("Vector scalar multiplication requires scalar")
        other_val = Decimal(str(other)) if not isinstance(other, Decimal) else other
        return Vector([x * other_val for x in self.data])

    def div(self, other: Decimal) -> Vector:
        """Divide by scalar."""
        if not _is_scalar(other):
            raise ToolExecutionError("Vector scalar division requires scalar")
        other_val = Decimal(str(other)) if not isinstance(other, Decimal) else other
        if _is_zero(other_val):
            raise ToolExecutionError("Division by zero")
        return Vector([x / other_val for x in self.data])


class Matrix:
    """Matrix class for Decimal mode linear algebra operations."""
    def __init__(self, data: list[list[float | Decimal]]) -> None:
        if not data or not all(isinstance(row, list) for row in data):
            raise ToolExecutionError("Matrix must be non-empty list of lists")
        row_len = len(data[0])
        if not all(len(row) == row_len for row in data):
            raise ToolExecutionError("All rows must have same length")
        if not all(_is_scalar(e) for row in data for e in row):
            raise ToolExecutionError("Matrix elements must be numbers")
        self.data = [
            [Decimal(str(e)) if not isinstance(e, Decimal) else e for e in row]
            for row in data
        ]

    def rows(self) -> int:
        """Return number of rows."""
        return len(self.data)

    def cols(self) -> int:
        """Return number of columns."""
        return len(self.data[0])

    def is_square(self) -> bool:
        """Check if matrix is square."""
        return self.rows() == self.cols()

    def transpose(self) -> Matrix:
        """Return transpose of matrix."""
        return Matrix([
            [self.data[r][c] for r in range(self.rows())]
            for c in range(self.cols())
        ])

    def trace(self) -> Decimal:
        """Return trace of matrix (sum of diagonal elements)."""
        if not self.is_square():
            raise ToolExecutionError("Trace requires square matrix")
        return sum(self.data[i][i] for i in range(self.rows()))

    def matmul(self, other: Matrix | Vector) -> Matrix | Vector:
        """Matrix multiplication with matrix or vector."""
        if isinstance(other, Vector):
            if self.cols() != other.dim():
                raise ToolExecutionError(
                    f"Dimension mismatch: matrix {self.rows()}x{self.cols()} "
                    f"vs vector {other.dim()}"
                )
            result = []
            for i in range(self.rows()):
                sum_val = Decimal(0)
                for j in range(self.cols()):
                    sum_val += self.data[i][j] * other.data[j]
                result.append(sum_val)
            return Vector(result)

        if not isinstance(other, Matrix):
            raise ToolExecutionError("Matrix multiplication requires matrix or vector")

        if self.cols() != other.rows():
            raise ToolExecutionError(
                f"Dimension mismatch: {self.rows()}x{self.cols()} "
                f"vs {other.rows()}x{other.cols()}"
            )

        result = [[Decimal(0) for _ in range(other.cols())] for _ in range(self.rows())]
        for i in range(self.rows()):
            for j in range(other.cols()):
                for k in range(self.cols()):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        return Matrix(result)

    def det(self) -> Decimal:
        """Return determinant using LU decomposition."""
        if not self.is_square():
            raise ToolExecutionError("Determinant requires square matrix")
        _, U = self._lu_decomposition()
        det = Decimal(1)
        for i in range(self.rows()):
            det *= U.data[i][i]
            if _is_zero(det):
                return Decimal(0)
        return det

    def inv(self) -> Matrix:
        """Return inverse using augmented matrix Gaussian elimination."""
        if not self.is_square():
            raise ToolExecutionError("Inverse requires square matrix")
        det = self.det()
        if _is_zero(det):
            raise ToolExecutionError("Matrix is singular (determinant is zero)")
        n = self.rows()
        I = Matrix([[Decimal(1) if i == j else Decimal(0) for j in range(n)] for i in range(n)])
        return self._solve_system(I)

    def rank(self) -> int:
        """Compute rank using row reduction."""
        n = self.rows()
        m = self.cols()

        working = [row[:] for row in self.data]
        rank = 0

        for col in range(min(n, m)):
            pivot_row = None
            for row in range(rank, n):
                if not _is_zero(working[row][col]):
                    pivot_row = row
                    break

            if pivot_row is None:
                continue

            working[rank], working[pivot_row] = working[pivot_row], working[rank]
            pivot = working[rank][col]

            for c in range(col, m):
                working[rank][c] = working[rank][c] / pivot

            for row in range(n):
                if row != rank and not _is_zero(working[row][col]):
                    factor = working[row][col]
                    for c in range(col, m):
                        working[row][c] = working[row][c] - factor * working[rank][c]

            rank += 1

        return rank

    def _lu_decomposition(self) -> tuple[Matrix, Matrix]:
        """Doolittle's LU decomposition without pivoting."""
        n = self.rows()
        L_data = [[Decimal(0) for _ in range(n)] for _ in range(n)]
        U_data = [[Decimal(0) for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for k in range(i, n):
                sum_val = sum(L_data[i][j] * U_data[j][k] for j in range(i))
                U_data[i][k] = self.data[i][k] - sum_val

            for k in range(i, n):
                if i == k:
                    L_data[i][i] = Decimal(1)
                else:
                    sum_val = sum(L_data[k][j] * U_data[j][i] for j in range(i))
                    if _is_zero(U_data[i][i]):
                        raise ToolExecutionError("Singular matrix in LU decomposition")
                    L_data[k][i] = (self.data[k][i] - sum_val) / U_data[i][i]

        return Matrix(L_data), Matrix(U_data)

    def _qr_decomposition(self) -> tuple[Matrix, Matrix]:
        """Gram-Schmidt QR decomposition."""
        n, m = self.rows(), self.cols()
        Q_data = [[Decimal(0) for _ in range(m)] for _ in range(n)]
        R_data = [[Decimal(0) for _ in range(m)] for _ in range(m)]

        for j in range(m):
            for i in range(n):
                Q_data[i][j] = self.data[i][j]

            for k in range(j):
                R_data[k][j] = sum(
                    Q_data[i][k] * self.data[i][j] for i in range(n)
                )
                for i in range(n):
                    Q_data[i][j] = Q_data[i][j] - R_data[k][j] * Q_data[i][k]

            R_data[j][j] = sum(Q_data[i][j] ** 2 for i in range(n)).sqrt()
            if _is_zero(R_data[j][j]):
                raise ToolExecutionError("Zero column in QR decomposition")

            for i in range(n):
                Q_data[i][j] = Q_data[i][j] / R_data[j][j]

        return Matrix(Q_data), Matrix(R_data)

    def _solve_system(self, B: Matrix | Vector) -> Matrix | Vector:
        """Solve AX = B using Gaussian elimination."""
        n = self.rows()
        if isinstance(B, Vector):
            B_data = [[x] for x in B.data]
        else:
            B_data = B.data

        augmented = [self.data[i] + B_data[i] for i in range(n)]

        for col in range(n):
            pivot_row = None
            for row in range(col, n):
                if not _is_zero(augmented[row][col]):
                    pivot_row = row
                    break

            if pivot_row is None:
                raise ToolExecutionError("Singular matrix in linear system")

            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
            pivot = augmented[col][col]

            num_cols = len(augmented[col])
            for c in range(col, num_cols):
                augmented[col][c] = augmented[col][c] / pivot

            for row in range(n):
                if row != col and not _is_zero(augmented[row][col]):
                    factor = augmented[row][col]
                    for c in range(col, num_cols):
                        augmented[row][c] = augmented[row][c] - factor * augmented[col][c]

        b_cols = num_cols - n
        solution = [augmented[i][n:] for i in range(n)]
        if isinstance(B, Vector):
            return Vector([row[0] for row in solution])
        return Matrix(solution)

    def add(self, other: Matrix | Decimal) -> Matrix:
        """Add matrix or scalar."""
        if _is_scalar(other):
            other_val = Decimal(str(other)) if not isinstance(other, Decimal) else other
            return Matrix([[x + other_val for x in row] for row in self.data])
        if isinstance(other, Matrix):
            if self.rows() != other.rows() or self.cols() != other.cols():
                raise ToolExecutionError("Dimension mismatch in matrix addition")
            return Matrix([
                [a + b for a, b in zip(row_a, row_b)]
                for row_a, row_b in zip(self.data, other.data)
            ])
        if not _is_matrix(other):
            raise ToolExecutionError("Matrix operation requires matrix or scalar")
        if self.rows() != len(other) or self.cols() != len(other[0]):
            raise ToolExecutionError("Dimension mismatch in matrix addition")
        other_dec = [
            [Decimal(str(e)) if not isinstance(e, Decimal) else e for e in row]
            for row in other
        ]
        return Matrix([
            [a + b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(self.data, other_dec)
        ])

    def sub(self, other: Matrix | Decimal) -> Matrix:
        """Subtract matrix or scalar."""
        if _is_scalar(other):
            other_val = Decimal(str(other)) if not isinstance(other, Decimal) else other
            return Matrix([[x - other_val for x in row] for row in self.data])
        if isinstance(other, Matrix):
            if self.rows() != other.rows() or self.cols() != other.cols():
                raise ToolExecutionError("Dimension mismatch in matrix subtraction")
            return Matrix([
                [a - b for a, b in zip(row_a, row_b)]
                for row_a, row_b in zip(self.data, other.data)
            ])
        if not _is_matrix(other):
            raise ToolExecutionError("Matrix operation requires matrix or scalar")
        if self.rows() != len(other) or self.cols() != len(other[0]):
            raise ToolExecutionError("Dimension mismatch in matrix subtraction")
        other_dec = [
            [Decimal(str(e)) if not isinstance(e, Decimal) else e for e in row]
            for row in other
        ]
        return Matrix([
            [a - b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(self.data, other_dec)
        ])

    def mul(self, other: Decimal) -> Matrix:
        """Multiply by scalar."""
        if not _is_scalar(other):
            raise ToolExecutionError("Matrix scalar multiplication requires scalar")
        other_val = Decimal(str(other)) if not isinstance(other, Decimal) else other
        return Matrix([[x * other_val for x in row] for row in self.data])

    def div(self, other: Decimal) -> Matrix:
        """Divide by scalar."""
        if not _is_scalar(other):
            raise ToolExecutionError("Matrix scalar division requires scalar")
        other_val = Decimal(str(other)) if not isinstance(other, Decimal) else other
        if _is_zero(other_val):
            raise ToolExecutionError("Division by zero")
        return Matrix([[x / other_val for x in row] for row in self.data])


def _norm_numpy(x: list[float]) -> float:
    """Compute norm using numpy."""
    import numpy as np
    arr = np.array(x, dtype=np.float64)
    return float(np.linalg.norm(arr))


def _dot_numpy(v1: list[float], v2: list[float]) -> float:
    """Compute dot product using numpy."""
    import numpy as np
    return float(np.dot(np.array(v1), np.array(v2)))


def _cross_numpy(v1: list[float], v2: list[float]) -> list[float]:
    """Compute cross product using numpy."""
    import numpy as np
    return [float(x) for x in np.cross(np.array(v1), np.array(v2))]


def _unit_numpy(v: list[float]) -> list[float]:
    """Compute unit vector using numpy."""
    import numpy as np
    arr = np.array(v, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < 1e-10:
        raise ToolExecutionError("Cannot normalize zero vector")
    return [float(x) for x in arr / norm]


def _angle_numpy(v1: list[float], v2: list[float]) -> float:
    """Compute angle between vectors using numpy."""
    import numpy as np
    a, b = np.array(v1), np.array(v2)
    dot = np.dot(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    if norms < 1e-10:
        raise ToolExecutionError("Cannot compute angle with zero vector")
    cos_val = max(-1.0, min(1.0, dot / norms))
    return float(math.acos(cos_val))


def _transpose_numpy(m: list[list[float]]) -> list[list[float]]:
    """Compute transpose using numpy."""
    import numpy as np
    return [[float(x) for x in row] for row in np.array(m).T.tolist()]


def _det_numpy(m: list[list[float]]) -> float:
    """Compute determinant using numpy."""
    import numpy as np
    det = np.linalg.det(np.array(m, dtype=np.float64))
    if abs(det) < 1e-10:
        return 0.0
    return float(det)


def _inv_numpy(m: list[list[float]]) -> list[list[float]]:
    """Compute inverse using numpy."""
    import numpy as np
    arr = np.array(m, dtype=np.float64)
    if abs(np.linalg.det(arr)) < 1e-10:
        raise ToolExecutionError("Matrix is singular (determinant is zero)")
    return [[float(x) for x in row] for row in np.linalg.inv(arr).tolist()]


def _trace_numpy(m: list[list[float]]) -> float:
    """Compute trace using numpy."""
    import numpy as np
    return float(np.trace(np.array(m, dtype=np.float64)))


def _rank_numpy(m: list[list[float]]) -> int:
    """Compute rank using numpy."""
    import numpy as np
    return int(np.linalg.matrix_rank(np.array(m, dtype=np.float64), tol=1e-10))


def _solve_numpy(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve linear system using numpy."""
    import numpy as np
    A_arr = np.array(A, dtype=np.float64)
    if abs(np.linalg.det(A_arr)) < 1e-10:
        raise ToolExecutionError("Singular matrix in linear system")
    return [float(x) for x in np.linalg.solve(A_arr, np.array(b))]


def _lu_numpy(m: list[list[float]]) -> list[list[list[float]]]:
    """Compute LU decomposition using numpy."""
    import numpy as np
    arr = np.array(m, dtype=np.float64)
    n = len(m)
    L_data = [[0.0] * n for _ in range(n)]
    U_data = [[0.0] * n for _ in range(n)]

    for i in range(n):
        L_data[i][i] = 1.0

        for k in range(i, n):
            U_data[i][k] = float(arr[i][k])
            for j in range(i):
                U_data[i][k] -= L_data[i][j] * U_data[j][k]

        for k in range(i + 1, n):
            L_data[k][i] = float(arr[k][i])
            for j in range(i):
                L_data[k][i] -= L_data[k][j] * U_data[j][i]
            if abs(U_data[i][i]) < 1e-10:
                raise ToolExecutionError("Singular matrix in LU decomposition")
            L_data[k][i] /= U_data[i][i]

    return [L_data, U_data]


def _qr_numpy(m: list[list[float]]) -> list[list[list[float]]]:
    """Compute QR decomposition using numpy."""
    import numpy as np
    Q, R = np.linalg.qr(np.array(m, dtype=np.float64))
    Q_list = [[float(x) for x in row] for row in Q.tolist()]
    R_list = [[float(x) for x in row] for row in R.tolist()]
    return [Q_list, R_list]


def _matmul_numpy(
    A: list[list[float]],
    B: list[list[float]] | list[float]
) -> list[list[float]] | list[float]:
    """Matrix multiplication using numpy."""
    import numpy as np
    if _is_vector(B):
        return [float(x) for x in np.array(A) @ np.array(B)]
    return [[float(x) for x in row] for row in (np.array(A) @ np.array(B)).tolist()]
