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

r"""Controlled State preparation.

This algorithm prepares a state $|\psi\rangle$ in a register initially at $|0\rangle$ by using
rotations $R_y$ for encoding amplitudes and $R_z$ for encoding phases.

Assume one wants to prepare the amplitude of a one qubit state

$$
\sqrt{p_0} |0\rangle + \sqrt{p_1} |1\rangle.
$$

This can be achieved by a rotation $R_y(\theta)$ where $\theta = \cos^{-1}(\sqrt{p_0})$.
For encoding the amplitude of a n-qubit quantum state one could use a similar approach to this, but
chaining conditional probabilities: first rotate qubit 1 by $\theta = \cos^{-1}(\sqrt{p_0})$, then
the second qubit by $\theta_0 = \cos^{-1}(\sqrt{p_{00}/p_{0}})$, conditioned on the first one being
in $|0\rangle$ and $\theta_1 = \cos^{-1}(\sqrt{p_{10}/p_{1}})$ conditioned by the first being in
$|1\rangle$, and so on. Here $p_y$ means the probability that the first len(y) qubits of the
original state are in the state $y$. Refer to equation (8) of [1] for the details.

This general scheme is handled by StatePreparationViaRotations. This class also uses
RotationTree to get the angles of rotation needed (which are converted to the value to be loaded
to the ROM to achieve such a rotation). RotationTree is a tree data structure which holds the
accumulated probability of each substring, i.e., the root holds the probability of measuring the
first qubit at 0, the branch1 node the probability of measuring the second qubit at 0 if the first
was measured at 1 and so on. The $2^i$ rotations needed to prepare the ith qubit are performed by
PRGAViaPhaseGradient. This essentially is a rotation gate array, that is, given a list of
angles it performs the kth rotation when the selection register is on state $|k\rangle$. This
rotation is done in the Z axis, but for encoding amplitude a rotation around Ry is needed, thus the
need of a $R_x(\pm \pi/2)$ gate before and after encoding the amplitudes of each qubit.

In order to perform the rotations as efficiently as possible, the angles are loaded into a register
(rot\_reg) which is added into a phase gradient. Then phase kickback causes an overall offset of
$e^{i2\pi x/2^b}$, where $x$ is the angle value loaded and $b$ the size of the rot\_reg. Below is an
example for rot\_reg\_size=2.

First there is the rot\_reg register with the value to be rotated (3 in this case) and the phase
gradient

$$
|3\rangle(e^{2\pi i 0/4}|0\rangle + e^{2\pi i 1/4}|1\rangle +
          e^{2\pi i 2/4}|2\rangle + e^{2\pi i 3/4}|3\rangle).
$$

Then the rot\_reg $|3\rangle$ register is added to the phase gradient and store the result in the
phase gradient register

$$
|3\rangle(e^{2\pi i 0/4}|3\rangle + e^{2\pi i 1/4}|0\rangle +
          e^{2\pi i 2/4}|1\rangle + e^{2\pi i 3/4}|2\rangle),
$$

but this is equivalent to the original state with a phase offset of $e^{2\pi i 1/4}$.


References:
    [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
    (https://arxiv.org/abs/1812.00954).
        Low, Kliuchnikov, Schaeffer. 2018.

"""

from typing import Callable, Dict, List, Tuple

import attrs
import numpy as np
from numpy.typing import ArrayLike
from typing_extensions import Self
import copy

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    GateWithRegisters,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.basic_gates.rotation import Rx
from qualtran.bloqs.qrom import QROM
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
from qualtran.bloqs.swap_network import CSwapApprox


@attrs.frozen
class StatePreparationViaRotations(GateWithRegisters):
    r"""Controlled state preparation without entangled residual using Ry and Rz rotations from [1].

    Given a quantum state of which the list of coefficients $c_i$ is known
    $$
        |\psi \rangle = \sum_{i=0}^{N-1}c_{i}|i\rangle
    $$
    this gate prepares $|\psi\rangle$ from $|0\rangle$ conditioned by a control qubit
    $$
        U((|0\rangle + |1\rangle)|0\rangle) = |0\rangle |0\rangle + |1\rangle |\psi\rangle.
    $$

    Args:
        phase_bitsize: size of the register that is used to store the rotation angles. Bigger values
            increase the accuracy of the results.
        state_coefficients: tuple of length 2^state_bitsizes that contains the complex coefficients of the state.
        control_bitsize: number of qubits of the control register. Set to zero for an uncontrolled gate.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    phase_bitsize: int
    state_coefficients: Tuple[complex, ...]
    control_bitsize: int = 0
    uncompute: bool = False

    def __attrs_post_init__(self):
        # a valid quantum state has a number of coefficients that is a power of two
        assert len(self.state_coefficients) == 2**self.state_bitsize
        # negative number of control bits is not allowed
        assert self.control_bitsize >= 0
        # the register to which the angle is written must be at least of size two
        assert self.phase_bitsize > 1
        # a valid quantum state must have norm one
        assert np.isclose(np.linalg.norm(self.state_coefficients), 1)

    @property
    def state_bitsize(self) -> int:
        return (len(self.state_coefficients) - 1).bit_length()

    @property
    def signature(self) -> Signature:
        return Signature.build(
            prepare_control=self.control_bitsize,
            target_state=self.state_bitsize,
            phase_gradient=self.phase_bitsize
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        r"""Parameters:
        * prepare_control: only if control_bitsize != 0
        * target_state: register where the state is written
        * phase_gradient: phase gradient state (will be left unaffected)
        """
        rotation_tree = RotationTree(self.state_coefficients, self.uncompute)
        amp_angles, phase_angles, _ = rotation_tree.get_angles()
        amp_reg = bb.allocate(self.phase_bitsize)
        ph_reg = bb.allocate(self.phase_bitsize)
        prev = ()
        for i in list(range(0, self.state_bitsize))[:: (1 - 2 * self.uncompute)]:
            soqs, prev, amp_reg, ph_reg = self._prepare_ith_qubit(
                bb, i, amp_angles[i], phase_angles[i], prev, amp_reg, ph_reg, **soqs
            )
        bb.free(amp_reg)
        bb.free(ph_reg)
        return soqs

    def _prepare_ith_qubit(
        self,
        bb: BloqBuilder,
        i: int,
        amp_angles: Tuple[float, ...],
        ph_angles: Tuple[float, ...],
        prev: Tuple[Tuple[int, ...], ...],
        amp_reg: SoquetT,
        ph_reg: SoquetT,
        **soqs: SoquetT,
    ):
        # if preparing the first qubit, load global phase, amplitudes and relative phase
        if self.uncompute:
            angles = (ph_angles, amp_angles)
            pre_post_gates = {1: (-1, Rx(angle=np.pi / 2), Rx(angle=-np.pi / 2))}
        else:
            angles = (amp_angles, ph_angles)
            pre_post_gates = {0: (-1, Rx(angle=np.pi / 2), Rx(angle=-np.pi / 2))}
        prga = PRGAViaPhaseGradient(
            i,
            self.phase_bitsize,
            angles,
            tuple(pre_post_gates.items()),
            self.control_bitsize + 1,
            first=(i == 0),
            # last=(i == self.state_bitsize - 1),
            last = False,
            prev_vals=prev
        )
        prga_soqs = {"target0_": amp_reg, "target1_":ph_reg}
        # prepare soquets for PRGA
        state_qubits = bb.split(soqs.pop("target_state"))
        if self.control_bitsize == 0:
            prga_soqs["control"] = state_qubits[i]
        elif self.control_bitsize == 1:
            prga_soqs["control"] = bb.join(np.array([soqs.pop("control"), state_qubits[i]]))
        else:
            ctrls = bb.split(soqs.pop("control"))
            prga_soqs["control"] = bb.join(np.array([*ctrls, state_qubits[i]]))
        if i != 0:
            prga_soqs["selection"] = bb.join(state_qubits[:i])
        prga_soqs["phase_gradient"] = soqs.pop("phase_gradient")
        # apply PRGA
        prga_soqs = bb.add_d(prga, **prga_soqs)
        # unprepare soquets for PRGA
        soqs["phase_gradient"] = prga_soqs.pop("phase_gradient")
        if self.control_bitsize == 0:
            state_qubits[i] = prga_soqs.pop("control")
        else:
            qubits = prga_soqs.pop("control")
            state_qubits[i] = qubits[-1]
            soqs["control"] = bb.join(qubits[:-1])
        if i != 0:
            state_qubits[:i] = bb.split(prga_soqs.pop("selection"))
        soqs["target_state"] = bb.join(state_qubits)
        return soqs, prga.effective_rom_values, prga_soqs.pop("target0_"), prga_soqs.pop("target1_")


@bloq_example
def _state_prep_via_rotation() -> StatePreparationViaRotations:
    state_coefs = (
        (-0.42677669529663675 - 0.1767766952966366j),
        (0.17677669529663664 - 0.4267766952966367j),
        (0.17677669529663675 - 0.1767766952966368j),
        (0.07322330470336305 - 0.07322330470336309j),
        (0.4267766952966366 - 0.17677669529663692j),
        (0.42677669529663664 + 0.17677669529663675j),
        (0.0732233047033631 + 0.17677669529663678j),
        (-0.07322330470336308 - 0.17677669529663678j),
    )
    state_prep_via_rotation = StatePreparationViaRotations(
        phase_bitsize=2, state_coefficients=state_coefs
    )
    return state_prep_via_rotation


_STATE_PREP_VIA_ROTATIONS_DOC = BloqDocSpec(
    bloq_cls=StatePreparationViaRotations,
    import_line='from qualtran.bloqs.state_preparation.state_preparation_via_rotation import StatePreparationViaRotations',
    examples=(_state_prep_via_rotation,),
)


@attrs.frozen
class PRGAViaPhaseGradient(Bloq):
    r"""Array of controlled rotations $Z^{\theta_i/2}$ for a list of angles $\theta$.

    It uses phase kickback and thus needs a phase gradient state in order to work. This
    state must be provided externally for efficiency, as it is unaffected and can thus be reused.
    Refer to [1], section on arbitrary quantum state preparation on page 3.

    Args:
        selection_bitsize: number of qubits used for encoding the selection register of the QROM, it
            must be equal to $\lceil \log_2(l_{rv}) \rceil$, where $l_{rv}$ is the number of angles
            provided.
        phase_bitsize: size of the register that is used to store the rotation angles. Bigger values
            increase the accuracy of the results.
        rom_values: the tuple of values to be loaded in the rom, which correspond to the angle of
            each rotation.
        control_bitsize: number of qubits of the control register. Set to zero for an uncontrolled gate.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    selection_bitsize: int
    phase_bitsize: int
    angles: Tuple[Tuple[float, ...], ...]
    # [int, [int, Bloq, Bloq]] -> key = row of angles that it applies to
    #                             values = (qubit of control to apply the gates to,
    #                                       gate to apply before the rotation,
    #                                       gate to apply after the rotation)
    pre_post_gates: Tuple[int, Tuple[int, Bloq, Bloq]]
    control_bitsize: int = 0
    first: bool = False
    last: bool = False
    prev_vals: Tuple[Tuple[int, ...], ...] = ()

    @property
    def signature(self) -> Signature:
        return Signature.build(
            control=self.control_bitsize,
            selection=self.selection_bitsize,
            **self.target_soqs,
            phase_gradient=self.phase_bitsize,
        )
    
    @property
    def target_soqs(self) -> Dict[str, int]:
        return dict([(f"target{i}_", self.phase_bitsize) for i in range(len(self.angles))])

    @property
    def effective_rom_values (self) -> Tuple[Tuple[int, ...], ...]:
        effective_rv = []
        for angle_list in self.angles:
            effective_rv.append(tuple([self.angle_to_rom_value(a) for a in angle_list]))
        return tuple(effective_rv)

    @property
    def rom_values(self) -> Tuple[Tuple[int, ...], ...]:
        rv_list = list(self.effective_rom_values)
        if not self.first:
            for i, rvs in enumerate(rv_list):
                rv_list[i] = tuple([rv ^ self.prev_vals[i][j // 2] for j, rv in enumerate(rvs)])
        return tuple(rv_list)

    @property
    def pre_post_all(self) -> Tuple[int]:
        indices = [ind for ind, _ in self.pre_post_gates]
        pre_post_all = []
        for i in range(len(self.angles)):
            if i in indices:
                pre_post_all.append(self.pre_post_gates[indices.index(i)][1])
            else:
                pre_post_all.append((0, None, None))
        return tuple(pre_post_all)

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        """Parameters:
        * control
        * selection (not necessary if selection_bitsize == 0)
        * qubit
        * target0_ ... targetN_ angle registers
        * phase_gradient
        """
        qrom = QROM(
            [np.array(rv) for rv in self.rom_values],
            selection_bitsizes=(self.selection_bitsize,),
            target_bitsizes=(self.phase_bitsize,) * len(self.rom_values),
        )
        phase_grad = soqs.pop("phase_gradient")
        if self.control_bitsize != 0:
            control = soqs.pop("control")
        # load angles in rot_reg (line 1 of eq (8) in [1])
        soqs = bb.add_d(qrom, **soqs)
        adder = AddIntoPhaseGrad(self.phase_bitsize, self.phase_bitsize)
        # phase kickback via phase_grad += rot_reg (line 2 of eq (8) in [1])
        for i in range(len(self.rom_values)):
            phase_grad, control, soqs = self.apply_ith_rotation(
                i, bb, adder, phase_grad, control, **soqs
            )
        # uncompute angle load in rot_reg to disentangle it from selection register
        # (line 1 of eq (8) in [1])
        if self.last:
            qrom = QROM(
                [np.array(rv) for rv in self.effective_rom_values],
                selection_bitsizes=(self.selection_bitsize,),
                target_bitsizes=(self.phase_bitsize,) * len(self.rom_values),
            )
            soqs = bb.add_d(qrom, **soqs)
        soqs["phase_gradient"] = phase_grad
        if self.control_bitsize != 0:
            soqs["control"] = control
        return soqs

    def apply_ith_rotation(
        self,
        i: int,
        bb: BloqBuilder,
        adder: Bloq,
        phase_grad: SoquetT,
        control: SoquetT,
        **soqs: SoquetT,
    ):
        pre_post = self.pre_post_all[i]
        controls = bb.split(control)
        if pre_post[1] is not None:
            controls[pre_post[0]] = bb.add(pre_post[1], q=controls[pre_post[0]])
        buffer = bb.allocate(self.phase_bitsize)
        controls[pre_post[0]], soqs[f"target{i}_"], buffer = bb.add(
            CSwapApprox(self.phase_bitsize),
            ctrl=controls[pre_post[0]],
            x=soqs[f"target{i}_"],
            y=buffer,
        )
        buffer, phase_grad = bb.add(adder, x=buffer, phase_grad=phase_grad)
        controls[pre_post[0]], soqs[f"target{i}_"], buffer = bb.add(
            CSwapApprox(self.phase_bitsize),
            ctrl=controls[pre_post[0]],
            x=soqs[f"target{i}_"],
            y=buffer,
        )
        if pre_post[2] is not None:
            controls[pre_post[0]] = bb.add(pre_post[2], q=controls[pre_post[0]])
        control = bb.join(controls)
        bb.free(buffer)
        return phase_grad, control, soqs

    def angle_to_rom_value(self, angle: float) -> int:
        r"""Given an angle, returns the value to be loaded in ROM.

        Returns the value to be loaded to a QROM to encode the given angle with a certain value of
        phase_bitsize.
        """
        rom_value_decimal = 2**self.phase_bitsize * angle / (2 * np.pi)
        return round(rom_value_decimal) % (2**self.phase_bitsize)


class RotationTree:
    r"""Used by `StatePreparationViaRotations` to get the corresponding rotation angles.

    The rotation angles are used to encode the amplitude of a state using the method described in
    [1], section on arbitrary quantum state preparation, page 3.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    def __init__(self, state: ArrayLike, uncompute: bool = False):
        self.state_bitsize = (len(state) - 1).bit_length()
        self._calc_amplitude_angles(state, uncompute)
        self._calc_phase_angles(state, uncompute)

    def get_angles(self) -> Tuple[List[List[int]], List[int]]:
        return tuple(self.amplitude_angles), tuple(self.phase_angles), self.global_phase

    def _calc_amplitude_angles(self, state: ArrayLike, uncompute: bool) -> None:
        r"""Gives a list of the ROM values to be loaded for preparing the amplitudes of a state.

        The ith element of the returned list is another list with the rom values to be loaded when
        preparing the amplitudes of the ith qubit for the given state.
        """
        slen = len(state)
        self.sum_total = np.zeros(2 * slen)
        for i in range(slen):
            self.sum_total[i + slen] = abs(state[i]) ** 2
        for i in range(slen - 1, 0, -1):
            self.sum_total[i] = self.sum_total[i << 1] + self.sum_total[(i << 1) | 1]
        self.amplitude_angles: List[List[int]] = []
        for i in range(self.state_bitsize):
            angles_this_layer: List[int] = []
            for node in range(1 << i, 1 << (i + 1)):
                angle = self._angle_0(node)
                if uncompute:
                    angle = 2 * np.pi - angle
                angles_this_layer.append(angle % (2 * np.pi))
            self.amplitude_angles.append(tuple(angles_this_layer))

    def phase_offsets(self) -> Tuple[float, ...]:
        sites = len(self.amplitude_angles)
        offs = np.zeros(2**sites)
        for i in range(sites):
            rang = 2 ** (sites - i)
            for block in range(2**i):
                offs[rang * block : rang * (block + 1)] += self.amplitude_angles[i][block] / 2
        return offs

    def _calc_phase_angles(self, state: ArrayLike, uncompute: bool) -> None:
        """Computes the rom value to be loaded to get the phase for each coefficient of the state.

        As we are using the equivalent to controlled Z to do the rotations instead of Rz, there
        is a phase offset for each coefficient that has to be corrected. This offset is half of the
        turn angle applied, and is added to the phase for each coefficient.
        """
        offsets = self.phase_offsets()
        angles = np.array([np.angle(c) for c in state])
        # flip angle if uncompute
        deltas = [(1 - 2 * uncompute) * (a - o) for a, o in zip(angles, offsets)]
        bitsize = (len(state) - 1).bit_length()
        for qi in range(bitsize):
            width = 2 ** (qi + 1)
            for split in range(2 ** (bitsize - qi - 1)):
                deltas[split * width + width // 2] -= deltas[split * width]
        deltas = [(d % (2 * np.pi)) for d in deltas]
        self.phase_angles = []
        for qi in range(bitsize):
            d_layer = [
                deltas[int(2 ** (bitsize - qi) * (split + 1 / 2))] for split in range(2**qi)
            ]
            self.phase_angles.append(tuple(d_layer))
        self.global_phase = deltas[0]

    def _angle_0(self, idx: int) -> float:
        r"""Angle that corresponds to p_0."""
        return 2 * np.arccos(np.sqrt(self._p0(idx)))

    def _p0(self, idx: int) -> float:
        if self.sum_total[idx] == 0:
            return 0
        return self.sum_total[idx << 1] / self.sum_total[idx]
