#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from functools import cached_property
from typing import Optional, Sequence, Tuple, Union

import cirq
import numpy as np
import pytest
import sympy
from attrs import define, field, frozen
from cirq.testing import random_unitary
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

from qualtran import Bloq, GateWithRegisters, Signature
from qualtran.bloqs.basic_gates.su2_rotation import SU2RotationGate
from qualtran.bloqs.generalized_qsp import (
    GeneralizedQSP,
    qsp_complementary_polynomial,
    qsp_phase_factors,
)
from qualtran.resource_counting import SympySymbolAllocator


def assert_angles_almost_equal(
    actual: Union[float, Sequence[float]], desired: Union[float, Sequence[float]]
):
    """Helper to check if two angle sequences are equal (up to multiples of 2*pi)"""
    np.testing.assert_almost_equal(np.exp(np.array(actual) * 1j), np.exp(np.array(desired) * 1j))


def check_polynomial_pair_on_random_points_on_unit_circle(
    P: Union[Sequence[complex], Polynomial],
    Q: Union[Sequence[complex], Polynomial],
    *,
    random_state: np.random.RandomState,
    n_points: int = 1000,
):
    P = Polynomial(P)
    Q = Polynomial(Q)

    for _ in range(n_points):
        z = np.exp(random_state.random() * np.pi * 2j)
        np.testing.assert_allclose(np.abs(P(z)) ** 2 + np.abs(Q(z)) ** 2, 1)


def random_qsp_polynomial(
    degree: int, *, random_state: np.random.RandomState, only_real_coeffs=False
) -> Sequence[complex]:
    poly = random_state.random(size=degree) / degree
    if not only_real_coeffs:
        poly = poly * np.exp(random_state.random(size=degree) * np.pi * 2j)
    return poly


@pytest.mark.parametrize("degree", [4, 5])
def test_complementary_polynomial_quick(degree: int):
    random_state = np.random.RandomState(42)

    for _ in range(2):
        P = random_qsp_polynomial(degree, random_state=random_state)
        Q = qsp_complementary_polynomial(P, verify=True)
        check_polynomial_pair_on_random_points_on_unit_circle(P, Q, random_state=random_state)


@pytest.mark.slow
@pytest.mark.parametrize("degree", [3, 4, 5, 10, 20, 30, 100])
def test_complementary_polynomial(degree: int):
    random_state = np.random.RandomState(42)

    for _ in range(10):
        P = random_qsp_polynomial(degree, random_state=random_state)
        Q = qsp_complementary_polynomial(P, verify=True)
        check_polynomial_pair_on_random_points_on_unit_circle(P, Q, random_state=random_state)


@pytest.mark.slow
@pytest.mark.parametrize("degree", [3, 4, 5, 10, 20, 30, 100])
def test_real_polynomial_has_real_complementary_polynomial(degree: int):
    random_state = np.random.RandomState(42)

    for _ in range(10):
        P = random_qsp_polynomial(degree, random_state=random_state, only_real_coeffs=True)
        Q = qsp_complementary_polynomial(P, verify=True)
        Q = np.around(Q, decimals=8)
        assert np.isreal(Q).all()


@frozen
class RandomGate(GateWithRegisters):
    bitsize: int
    matrix: Tuple[Tuple[complex, ...], ...] = field(
        converter=lambda mat: tuple(tuple(row) for row in mat)
    )

    @staticmethod
    def create(bitsize: int, *, random_state=None) -> 'RandomGate':
        matrix = random_unitary(2**bitsize, random_state=random_state)
        return RandomGate(bitsize, matrix)

    @property
    def signature(self) -> Signature:
        return Signature.build(q=self.bitsize)

    def _unitary_(self):
        return np.array(self.matrix)

    def adjoint(self) -> 'RandomGate':
        return RandomGate(self.bitsize, np.conj(self.matrix).T)

    def __pow__(self, power):
        if power == -1:
            return self.adjoint()
        return NotImplemented


def evaluate_polynomial_of_matrix(
    P: Sequence[complex], U: NDArray, *, negative_power: int = 0
) -> NDArray:
    assert U.ndim == 2 and U.shape[0] == U.shape[1]

    pow_U = np.linalg.matrix_power(U.conj().T, negative_power)
    result = np.zeros(U.shape, dtype=U.dtype)

    for c in P:
        result += pow_U * c
        pow_U = pow_U @ U

    return result


def assert_matrices_almost_equal(A: NDArray, B: NDArray):
    assert A.shape == B.shape
    assert np.linalg.norm(A - B) <= 1e-5


def verify_generalized_qsp(
    U: GateWithRegisters,
    P: Sequence[complex],
    Q: Optional[Sequence[complex]] = None,
    *,
    negative_power: int = 0,
):
    input_unitary = cirq.unitary(U)
    N = input_unitary.shape[0]
    if Q is None:
        gqsp_U = GeneralizedQSP.from_qsp_polynomial(U, P, negative_power=negative_power)
    else:
        gqsp_U = GeneralizedQSP(U, P, Q, negative_power=negative_power)
    result_unitary = cirq.unitary(gqsp_U)

    expected_top_left = evaluate_polynomial_of_matrix(
        P, input_unitary, negative_power=negative_power
    )
    actual_top_left = result_unitary[:N, :N]
    assert_matrices_almost_equal(expected_top_left, actual_top_left)

    expected_bottom_left = evaluate_polynomial_of_matrix(
        gqsp_U.Q, input_unitary, negative_power=negative_power
    )
    actual_bottom_left = result_unitary[N:, :N]
    assert_matrices_almost_equal(expected_bottom_left, actual_bottom_left)


@pytest.mark.slow
@pytest.mark.parametrize("bitsize", [1, 2, 3])
@pytest.mark.parametrize("degree", [2, 3, 4, 5, 50, 100, 150, 180])
def test_generalized_qsp_with_real_poly_on_random_unitaries(bitsize: int, degree: int):
    random_state = np.random.RandomState(42)

    for _ in range(10):
        U = RandomGate.create(bitsize, random_state=random_state)
        P = random_qsp_polynomial(degree, random_state=random_state, only_real_coeffs=True)
        verify_generalized_qsp(U, P)


@pytest.mark.slow
@pytest.mark.parametrize("bitsize", [1, 2, 3])
@pytest.mark.parametrize("degree", [2, 3, 4, 5, 50, 100, 120])
@pytest.mark.parametrize("negative_power", [0, 1, 2])
def test_generalized_qsp_with_complex_poly_on_random_unitaries(
    bitsize: int, degree: int, negative_power: int
):
    random_state = np.random.RandomState(42)

    for _ in range(10):
        U = RandomGate.create(bitsize, random_state=random_state)
        P = random_qsp_polynomial(degree, random_state=random_state)
        verify_generalized_qsp(U, P, negative_power=negative_power)


@pytest.mark.parametrize("negative_power", [0, 1, 2])
def test_call_graph(negative_power: int):
    random_state = np.random.RandomState(42)

    ssa = SympySymbolAllocator()
    theta = ssa.new_symbol("theta")
    phi = ssa.new_symbol("phi")
    lambd = ssa.new_symbol("lambda")
    arbitrary_rotation = SU2RotationGate(theta, phi, lambd)

    def catch_rotations(bloq: Bloq) -> Bloq:
        if isinstance(bloq, SU2RotationGate):
            return arbitrary_rotation
        return bloq

    U = RandomGate.create(1, random_state=random_state)
    P = (0.5, 0, 0.5)
    gsqp_U = GeneralizedQSP.from_qsp_polynomial(U, P, negative_power=negative_power)

    g, sigma = gsqp_U.call_graph(max_depth=1, generalizer=catch_rotations)

    expected_counts = {U.controlled(control_values=[0]): 3 - negative_power, arbitrary_rotation: 3}
    if negative_power > 0:
        expected_counts[U.adjoint().controlled()] = negative_power

    assert sigma == expected_counts


@define(slots=False)
class SymbolicGQSP:
    r"""Run the Generalized QSP algorithm on a symbolic input unitary

    Assuming the input unitary $U$ is symbolic, we construct the signal matrix $A$ as:

        $$
        A = \begin{bmatrix} U & 0 \\ 0 & 1 \end{bmatrix}
        $$

    This matrix $A$ is used to symbolically compute the GQSP transformed unitary, and the
    norms of the output blocks are upper-bounded. We then check if this upper-bound is negligible.
    """

    P: Sequence[complex]

    @cached_property
    def Q(self):
        return qsp_complementary_polynomial(self.P)

    def get_symbolic_qsp_matrix(self, U: sympy.Symbol):
        theta, phi, lambd = qsp_phase_factors(self.P, self.Q)

        # 0-controlled U gate
        A = np.array([[U, 0], [0, 1]])

        # compute the final GQSP unitary
        # mirrors the implementation of GeneralizedQSP::decompose_from_registers,
        # as cirq.unitary does not support symbolic matrices
        W = SU2RotationGate(theta[0], phi[0], lambd).rotation_matrix
        for t, p in zip(theta[1:], phi[1:]):
            W = A @ W
            W = SU2RotationGate(t, p, 0).rotation_matrix @ W

        return W

    @staticmethod
    def upperbound_matrix_norm(M, U) -> float:
        return float(sum(abs(c) for c in sympy.Poly(M, U).coeffs()))

    def verify(self):
        check_polynomial_pair_on_random_points_on_unit_circle(
            self.P, self.Q, random_state=np.random.RandomState(42)
        )
        U = sympy.symbols('U')
        W = self.get_symbolic_qsp_matrix(U)
        actual_PU, actual_QU = W[:, 0]

        expected_PU = sympy.Poly(reversed(self.P), U)
        expected_QU = sympy.Poly(reversed(self.Q), U)

        error_PU = self.upperbound_matrix_norm(expected_PU - actual_PU, U)
        error_QU = self.upperbound_matrix_norm(expected_QU - actual_QU, U)

        assert abs(error_PU) <= 1e-5
        assert abs(error_QU) <= 1e-5


@pytest.mark.slow
@pytest.mark.parametrize("degree", [2, 3, 4, 5, 10])
def test_generalized_real_qsp_with_symbolic_signal_matrix(degree: int):
    random_state = np.random.RandomState(102)

    for _ in range(10):
        P = random_qsp_polynomial(degree, random_state=random_state)
        SymbolicGQSP(P).verify()
