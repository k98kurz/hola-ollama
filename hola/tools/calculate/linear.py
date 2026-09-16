"""
Linear algebra support for the calculate tool.
Supports vectors, matrices, and linear algebra operations with both
Decimal (precision mode) and NumPy (float mode) implementations.
"""

from __future__ import annotations
from decimal import Decimal
from typing import Any
from hola.errors import ToolExecutionError
import math


_numpy_mode: bool | None = None


try:
    import numpy as np
except ImportError:
    _numpy_mode = False

def set_numpy_mode(force: bool | None) -> None:
    """Set numpy mode for testing. None=auto, True=force numpy,
        False=force custom.
    """
    global _numpy_mode
    _numpy_mode = force


def _use_numpy(precision: int) -> bool:
    """Check if numpy should be used based on precision and mode setting."""
    try:
        return precision == 0 and _numpy_mode is not False
    except ImportError:
        return False


def _use_float_mode(precision: int) -> bool:
    """Check if float fallback mode should be used."""
    return precision == 0 and not _use_numpy(precision)


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


def _to_dec(value: float | Decimal) -> Decimal:
    """Convert a scalar to `Decimal` if it is not already one."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


class Vector:
    """Vector class for Decimal mode linear algebra operations."""
    def __init__(self, data: list[float | Decimal]) -> None:
        if not data or not all(_is_scalar(e) for e in data):
            raise ToolExecutionError("Vector must be non-empty list of numbers")
        self.data = [_to_dec(e) for e in data]

    def dim(self) -> int:
        """Return vector dimension."""
        return len(self.data)

    def norm(self) -> Decimal:
        """Return Euclidean norm (L2 norm)."""
        return sum(x * x for x in self.data).sqrt()

    def dot(self, other: Vector) -> Decimal:
        """Return dot product with another vector."""
        if self.dim() != other.dim():
            raise ToolExecutionError(
                f"Dimension mismatch: {self.dim()} vs {other.dim()}"
            )
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
            other_val = _to_dec(other)
            return Vector([x + other_val for x in self.data])
        if not _is_vector(other):
            raise ToolExecutionError("Vector operation requires vector or scalar")
        if self.dim() != len(other):
            raise ToolExecutionError(
                f"Dimension mismatch: {self.dim()} vs {len(other)}"
            )
        other_dec = [_to_dec(e) for e in other]
        return Vector([a + b for a, b in zip(self.data, other_dec)])

    def sub(self, other: Vector | Decimal) -> Vector:
        """Subtract vector or scalar."""
        if _is_scalar(other):
            other_val = _to_dec(other)
            return Vector([x - other_val for x in self.data])
        if not _is_vector(other):
            raise ToolExecutionError("Vector operation requires vector or scalar")
        if self.dim() != len(other):
            raise ToolExecutionError(
                f"Dimension mismatch: {self.dim()} vs {len(other)}"
            )
        other_dec = [_to_dec(e) for e in other]
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
            raise ToolExecutionError(
                "Matrix multiplication requires matrix or vector"
            )

        if self.cols() != other.rows():
            raise ToolExecutionError(
                f"Dimension mismatch: {self.rows()}x{self.cols()} "
                f"vs {other.rows()}x{other.cols()}"
            )

        result = [
            [Decimal(0) for _ in range(other.cols())]
            for _ in range(self.rows())
        ]
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
        I = Matrix([
            [Decimal(1) if i == j else Decimal(0) for j in range(n)]
            for i in range(n)
        ])
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
                        raise ToolExecutionError(
                            "Singular matrix in LU decomposition"
                        )
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

            temp = augmented[col]
            augmented[col] = augmented[pivot_row]
            augmented[pivot_row] = temp
            pivot = augmented[col][col]

            num_cols = len(augmented[col])
            for c in range(col, num_cols):
                augmented[col][c] = augmented[col][c] / pivot

            for row in range(n):
                if row != col and not _is_zero(augmented[row][col]):
                    factor = augmented[row][col]
                    for c in range(col, num_cols):
                        augmented[row][c] -= factor * augmented[col][c]

        b_cols = num_cols - n
        solution = [augmented[i][n:] for i in range(n)]
        if isinstance(B, Vector):
            return Vector([row[0] for row in solution])
        return Matrix(solution)

    def add(self, other: Matrix | Decimal) -> Matrix:
        """Add matrix or scalar."""
        if _is_scalar(other):
            other_val = _to_dec(other)
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
            other_val = _to_dec(other)
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
    arr = np.array(x, dtype=np.float64)
    return float(np.linalg.norm(arr))


def _dot_numpy(v1: list[float], v2: list[float]) -> float:
    """Compute dot product using numpy."""
    return float(np.dot(np.array(v1), np.array(v2)))


def _cross_numpy(v1: list[float], v2: list[float]) -> list[float]:
    """Compute cross product using numpy."""
    return [float(x) for x in np.cross(np.array(v1), np.array(v2))]


def _unit_numpy(v: list[float]) -> list[float]:
    """Compute unit vector using numpy."""
    arr = np.array(v, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < 1e-10:
        raise ToolExecutionError("Cannot normalize zero vector")
    return [float(x) for x in arr / norm]


def _angle_numpy(v1: list[float], v2: list[float]) -> float:
    """Compute angle between vectors using numpy."""
    a, b = np.array(v1), np.array(v2)
    dot = np.dot(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    if norms < 1e-10:
        raise ToolExecutionError("Cannot compute angle with zero vector")
    cos_val = max(-1.0, min(1.0, dot / norms))
    return float(math.acos(cos_val))


def _transpose_numpy(m: list[list[float]]) -> list[list[float]]:
    """Compute transpose using numpy."""
    return [[float(x) for x in row] for row in np.array(m).T.tolist()]


def _det_numpy(m: list[list[float]]) -> float:
    """Compute determinant using numpy."""
    det = np.linalg.det(np.array(m, dtype=np.float64))
    if abs(det) < 1e-10:
        return 0.0
    return float(det)


def _inv_numpy(m: list[list[float]]) -> list[list[float]]:
    """Compute inverse using numpy."""
    arr = np.array(m, dtype=np.float64)
    if abs(np.linalg.det(arr)) < 1e-10:
        raise ToolExecutionError("Matrix is singular (determinant is zero)")
    return [[float(x) for x in row] for row in np.linalg.inv(arr).tolist()]


def _trace_numpy(m: list[list[float]]) -> float:
    """Compute trace using numpy."""
    return float(np.trace(np.array(m, dtype=np.float64)))


def _rank_numpy(m: list[list[float]]) -> int:
    """Compute rank using numpy."""
    return int(np.linalg.matrix_rank(np.array(m, dtype=np.float64), tol=1e-10))


def _solve_numpy(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve linear system using numpy."""
    A_arr = np.array(A, dtype=np.float64)
    if abs(np.linalg.det(A_arr)) < 1e-10:
        raise ToolExecutionError("Singular matrix in linear system")
    return [float(x) for x in np.linalg.solve(A_arr, np.array(b))]


def _lu_numpy(m: list[list[float]]) -> list[list[list[float]]]:
    """Compute LU decomposition using numpy."""
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
    Q, R = np.linalg.qr(np.array(m, dtype=np.float64))
    Q_list = [[float(x) for x in row] for row in Q.tolist()]
    R_list = [[float(x) for x in row] for row in R.tolist()]
    return [Q_list, R_list]


def _matmul_numpy(
    A: list[list[float]],
    B: list[list[float]] | list[float]
) -> list[list[float]] | list[float]:
    """Matrix multiplication using numpy."""
    if _is_vector(B):
        return [float(x) for x in np.array(A) @ np.array(B)]
    return [[float(x) for x in row] for row in (np.array(A) @ np.array(B)).tolist()]


def _norm_float(x: list[float]) -> float:
    """Compute norm using pure Python float."""
    return math.sqrt(sum(v * v for v in x))


def _dot_float(v1: list[float], v2: list[float]) -> float:
    """Compute dot product using pure Python float."""
    return sum(a * b for a, b in zip(v1, v2))


def _cross_float(v1: list[float], v2: list[float]) -> list[float]:
    """Compute cross product using pure Python float."""
    if len(v1) != 3 or len(v2) != 3:
        raise ToolExecutionError("Cross product requires 3D vectors")
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]


def _unit_float(v: list[float]) -> list[float]:
    """Compute unit vector using pure Python float."""
    norm = _norm_float(v)
    if norm < 1e-10:
        raise ToolExecutionError("Cannot normalize zero vector")
    return [x / norm for x in v]


def _angle_float(v1: list[float], v2: list[float]) -> float:
    """Compute angle between vectors using pure Python float."""
    dot_val = _dot_float(v1, v2)
    norms = _norm_float(v1) * _norm_float(v2)
    if norms < 1e-10:
        raise ToolExecutionError("Cannot compute angle with zero vector")
    cos_val = max(-1.0, min(1.0, dot_val / norms))
    return math.acos(cos_val)


def _transpose_float(m: list[list[float]]) -> list[list[float]]:
    """Compute transpose using pure Python float."""
    return [[m[r][c] for r in range(len(m))] for c in range(len(m[0]))]


def _det_float(m: list[list[float]]) -> float:
    """Compute determinant using Gaussian elimination."""
    n = len(m)
    det = 1.0
    m_copy = [[float(x) for x in row] for row in m]

    for i in range(n):
        pivot_row = i
        for r in range(i, n):
            if abs(m_copy[r][i]) > abs(m_copy[pivot_row][i]):
                pivot_row = r
        if abs(m_copy[pivot_row][i]) < 1e-10:
            return 0.0
        if pivot_row != i:
            m_copy[i], m_copy[pivot_row] = m_copy[pivot_row], m_copy[i]
            det *= -1
        det *= m_copy[i][i]
        pivot = m_copy[i][i]
        for r in range(i + 1, n):
            factor = m_copy[r][i] / pivot
            for c in range(i, n):
                m_copy[r][c] -= factor * m_copy[i][c]
    return det


def _trace_float(m: list[list[float]]) -> float:
    """Compute trace using pure Python float."""
    if len(m) != len(m[0]):
        raise ToolExecutionError("Trace requires square matrix")
    return sum(m[i][i] for i in range(len(m)))


def _rank_float(m: list[list[float]]) -> int:
    """Compute rank using Gaussian elimination."""
    n_rows, n_cols = len(m), len(m[0])
    m_copy = [[float(x) for x in row] for row in m]
    rank = 0
    row = 0
    for col in range(n_cols):
        pivot_row = None
        for r in range(row, n_rows):
            if abs(m_copy[r][col]) > 1e-10:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        m_copy[row], m_copy[pivot_row] = m_copy[pivot_row], m_copy[row]
        pivot = m_copy[row][col]
        for c in range(col, n_cols):
            m_copy[row][c] /= pivot
        for r in range(n_rows):
            if r != row and abs(m_copy[r][col]) > 1e-10:
                factor = m_copy[r][col]
                for c in range(col, n_cols):
                    m_copy[r][c] -= factor * m_copy[row][c]
        row += 1
        rank += 1
        if row >= n_rows:
            break
    return rank


def _inv_float(m: list[list[float]]) -> list[list[float]]:
    """Compute inverse using Gaussian elimination."""
    n = len(m)
    if len(m[0]) != n:
        raise ToolExecutionError("Inverse requires square matrix")
    if abs(_det_float(m)) < 1e-10:
        raise ToolExecutionError("Matrix is singular (determinant is zero)")
    aug = [
        [float(m[i][j]) for j in range(n)]
        + [1.0 if j == i else 0.0 for j in range(n)]
        for i in range(n)
    ]
    for i in range(n):
        pivot = aug[i][i]
        if abs(pivot) < 1e-10:
            raise ToolExecutionError("Matrix is singular")
        for j in range(2 * n):
            aug[i][j] /= pivot
        for r in range(n):
            if r != i:
                factor = aug[r][i]
                for j in range(2 * n):
                    aug[r][j] -= factor * aug[i][j]
    return [[aug[i][j + n] for j in range(n)] for i in range(n)]


def _solve_float(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve linear system using Gaussian elimination."""
    n = len(A)
    if len(A[0]) != n or len(b) != n:
        raise ToolExecutionError("Linear system requires square matrix")
    if abs(_det_float(A)) < 1e-10:
        raise ToolExecutionError("Singular matrix in linear system")
    aug = [[float(A[i][j]) for j in range(n)] + [float(b[i])] for i in range(n)]
    for i in range(n):
        pivot = aug[i][i]
        if abs(pivot) < 1e-10:
            raise ToolExecutionError("Singular matrix")
        for j in range(i, n + 1):
            aug[i][j] /= pivot
        for r in range(i + 1, n):
            factor = aug[r][i]
            for j in range(i, n + 1):
                aug[r][j] -= factor * aug[i][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = aug[i][n] - sum(aug[i][j] * x[j] for j in range(i + 1, n))
    return x


def _lu_float(m: list[list[float]]) -> list[list[list[float]]]:
    """Compute LU decomposition using pure Python float."""
    n = len(m)
    L_data = [[0.0] * n for _ in range(n)]
    U_data = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L_data[i][i] = 1.0
        for k in range(i, n):
            U_data[i][k] = float(m[i][k])
            for j in range(i):
                U_data[i][k] -= L_data[i][j] * U_data[j][k]
        for k in range(i + 1, n):
            L_data[k][i] = float(m[k][i])
            for j in range(i):
                L_data[k][i] -= L_data[k][j] * U_data[j][i]
            if abs(U_data[i][i]) < 1e-10:
                raise ToolExecutionError("Singular matrix in LU decomposition")
            L_data[k][i] /= U_data[i][i]
    return [L_data, U_data]


def _qr_float(m: list[list[float]]) -> list[list[list[float]]]:
    """Compute QR decomposition using Gram-Schmidt."""
    n_rows, n_cols = len(m), len(m[0])
    cols = [[float(m[r][c]) for r in range(n_rows)] for c in range(n_cols)]
    Q_cols = []
    R_data = [[0.0] * n_cols for _ in range(n_cols)]

    for j in range(n_cols):
        v = cols[j][:]
        for i in range(j):
            R_data[i][j] = _dot_float(Q_cols[i], v)
            v = [v[k] - R_data[i][j] * Q_cols[i][k] for k in range(n_rows)]
        R_data[j][j] = _norm_float(v)
        if R_data[j][j] < 1e-10:
            raise ToolExecutionError("QR decomposition failed")
        Q_cols.append([v[k] / R_data[j][j] for k in range(n_rows)])

    Q_data = [[Q_cols[c][r] for c in range(n_cols)] for r in range(n_rows)]
    Q_normalized = [[-Q_data[i][j] for j in range(n_cols)] for i in range(n_rows)]
    R_normalized = [[-R_data[i][j] for j in range(n_cols)] for i in range(n_cols)]
    return [Q_normalized, R_normalized]


def _matmul_float(
    A: list[list[float]],
    B: list[list[float]] | list[float]
) -> list[list[float]] | list[float]:
    """Matrix multiplication using pure Python float."""
    if _is_vector(B):
        B_col = [[float(b)] for b in B]
        result = [[0.0] for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B)):
                result[i][0] += A[i][j] * B_col[j][0]
        return [result[i][0] for i in range(len(A))]
    n = len(A)
    m = len(B)
    p = len(B[0])
    result = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            for k in range(m):
                result[i][j] += A[i][k] * B[k][j]
    return result


def _add_vectors_float(v1: list[float], v2: list[float]) -> list[float]:
    """Add two vectors using pure Python float."""
    return [a + b for a, b in zip(v1, v2)]


def _add_matrices_float(
        m1: list[list[float]], m2: list[list[float]],
    ) -> list[list[float]]:
    """Add two matrices using pure Python float."""
    return [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(m1, m2)]


def _add_vector_scalar_float(v: list[float], s: float) -> list[float]:
    """Add scalar to vector using pure Python float."""
    return [x + s for x in v]


def _add_matrix_scalar_float(m: list[list[float]], s: float) -> list[list[float]]:
    """Add scalar to matrix using pure Python float."""
    return [[x + s for x in row] for row in m]


def _sub_vectors_float(v1: list[float], v2: list[float]) -> list[float]:
    """Subtract two vectors using pure Python float."""
    return [a - b for a, b in zip(v1, v2)]


def _sub_matrices_float(
        m1: list[list[float]], m2: list[list[float]],
    ) -> list[list[float]]:
    """Subtract two matrices using pure Python float."""
    return [[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(m1, m2)]


def _sub_vector_scalar_float(v: list[float], s: float) -> list[float]:
    """Subtract scalar from vector using pure Python float."""
    return [x - s for x in v]


def _sub_matrix_scalar_float(m: list[list[float]], s: float) -> list[list[float]]:
    """Subtract scalar from matrix using pure Python float."""
    return [[x - s for x in row] for row in m]


def _mul_vector_scalar_float(v: list[float], s: float) -> list[float]:
    """Multiply vector by scalar using pure Python float."""
    return [x * s for x in v]


def _mul_matrix_scalar_float(m: list[list[float]], s: float) -> list[list[float]]:
    """Multiply matrix by scalar using pure Python float."""
    return [[x * s for x in row] for row in m]


def _div_vector_scalar_float(v: list[float], s: float) -> list[float]:
    """Divide vector by scalar using pure Python float."""
    if abs(s) < 1e-10:
        raise ToolExecutionError("Division by zero")
    return [x / s for x in v]


def _div_matrix_scalar_float(m: list[list[float]], s: float) -> list[list[float]]:
    """Divide matrix by scalar using pure Python float."""
    if abs(s) < 1e-10:
        raise ToolExecutionError("Division by zero")
    return [[x / s for x in row] for row in m]
