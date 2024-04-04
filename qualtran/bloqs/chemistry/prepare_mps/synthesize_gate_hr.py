from typing import Dict, List, Tuple

import attrs
import cirq
import numpy as np

from qualtran import Bloq, BloqBuilder, Signature, SoquetT
from qualtran.bloqs.basic_gates import CNOT, Hadamard, XGate, ZGate
from qualtran.bloqs.mcmt.multi_control_multi_target_pauli import MultiControlPauli
from qualtran.bloqs.qrom import QROM
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad, PhaseGradientState
from qualtran.bloqs.swap_network import CSwapApprox
from qualtran.bloqs.state_preparation.state_preparation_via_rotation import RotationTree
from qualtran.bloqs.basic_gates.rotation import Rx


@attrs.frozen
class SynthesizeGateHR(Bloq):

    cols: Tuple[Tuple[int, Tuple[complex, ...]], ...]
    phase_bitsize: int
    internal_phase_grad: bool = False
    internal_refl_ancilla: bool = True

    @property
    def gate_bitsize(self):
        return (len(self.cols[0][1]) - 1).bit_length()

    @property
    def signature(self) -> Signature:
        return Signature.build(
            reflection_ancilla=(not self.internal_refl_ancilla),
            state=self.gate_bitsize,
            phase_grad=self.phase_bitsize * (not self.internal_phase_grad),
        )

    def build_composite_bloq(
        self, bb: BloqBuilder, **soqs: SoquetT
    ) -> Dict[str, SoquetT]:
        if self.internal_refl_ancilla:
            soqs["ra"] = bb.allocate(1)
        else:
            soqs["ra"] = soqs.pop("reflection_ancilla")
        if self.internal_phase_grad:
            soqs["phase_grad"] = bb.add(PhaseGradientState(self.phase_bitsize))

        soqs["ph_reg"] = bb.allocate(self.phase_bitsize)
        soqs["amp_reg"] = bb.allocate(self.phase_bitsize)
        soqs["ra"] = bb.add(XGate(), q=soqs["ra"])
        for i, ui in self.cols:
            amps, phases, _ = RotationTree(ui, False).get_angles()
            amps_t, phases_t, _ = RotationTree(ui, True).get_angles()
            # soqs = self._r_ui_t(bb, amps_t, phases_t, **soqs)
            # soqs["ra"], soqs["state"] = self._reflection_core(bb, i, soqs["ra"], soqs["state"])
            soqs = self._r_ui(bb, amps, phases, **soqs)
        bb.free(soqs.pop("ph_reg"))
        bb.free(soqs.pop("amp_reg"))

        if self.internal_phase_grad:
            bb.add(PhaseGradientState(self.phase_bitsize).adjoint(), phase_grad=soqs.pop("phase_grad"))
        if self.internal_refl_ancilla:
            soqs["ra"] = bb.add(XGate(), q=soqs["ra"])
            bb.free(soqs.pop("ra"))
        else:
            soqs["reflection_ancilla"] = soqs.pop("ra")
        return soqs


    def _r_ui_t(self, bb: BloqBuilder, amps_t: Tuple[Tuple[float,...],...], phases_t: Tuple[Tuple[float,...],...], **soqs: SoquetT) -> Dict[str, SoquetT]:
        r"""Registers are:
            * ra: reflection ancilla
            * state: input qubits
            * amp_reg: register to store the amplitude angle
            * ph_reg: register to store the phase angle
            * phase_grad: phase gradient register
        """
        state = bb.split(soqs.pop("state"))
        qubit = state[-1]
        select = bb.join(state[::-1])
        initial_qrom = DifferentialQROM(((0,),(0,)), (phases_t[-1], amps_t[-1]), self.phase_bitsize)
        qubit, select, amp_reg, ph_reg = bb.add()
        soqs["state"] = bb.join(state)
        return soqs

    def _r_ui(self, bb: BloqBuilder, amps: Tuple[Tuple[float,...],...], phases: Tuple[Tuple[float,...],...], **soqs: SoquetT) -> Dict[str, SoquetT]:
        r"""Registers are:
            * ra: reflection ancilla
            * state: input qubits
            * amp_reg: register to store the amplitude angle
            * ph_reg: register to store the phase angle
            * phase_grad: phase gradient register
        """
        print(amps, phases)
        pre_post_gates = ((0, (-1, Rx(angle=np.pi / 2), Rx(angle=-np.pi / 2))),)
        qubits = bb.split(soqs.pop("state"))
        soqs_qrom = {"target0_": soqs.pop("amp_reg"), "target1_": soqs.pop("ph_reg")}
        for i, (amp_qubit, ph_qubit) in enumerate(zip(amps, phases)):
            if i == 0:
                prev_angs = ((0,),(0,))
            else:
                prev_angs = (amps[i-1], phases[i-1])
            qrom = DifferentialQROM(prev_angs, (amp_qubit, ph_qubit), self.phase_bitsize)
            adders = RotationViaAddition(pre_post_gates, 2, self.phase_bitsize, 2)
            if i != 0:
                soqs_qrom["selection"] = bb.join(qubits[:i])
            soqs_qrom = bb.add_d(qrom, **soqs_qrom)
            if i != 0:
                qubits[:i] = soqs_qrom.pop("selection")
            control = bb.join(np.array([soqs.pop("ra"), qubits[i]]))
            control, soqs_qrom["target0_"], soqs_qrom["target1_"], soqs["phase_grad"] = bb.add(
                adders, control=control, target0_=soqs_qrom["target0_"], target1_=soqs_qrom["target1_"], phase_grad=soqs["phase_grad"]
            )
            control_qubits = bb.split(control)
            soqs["ra"] = control_qubits[0]
            qubits[i] = control_qubits[1]
        qrom = DifferentialQROM((amps[-1], phases[-1]), ((0,), (0,)), self.phase_bitsize)
        if i != 0:
            soqs_qrom["selection"] = bb.join(qubits[:-1])
        soqs_qrom = bb.add_d(qrom, **soqs_qrom)
        if i != 0:
            qubits[:-1] = soqs_qrom.pop("selection")
        return {"state": bb.join(qubits), "amp_reg": soqs_qrom.pop("target0_"), "ph_reg": soqs_qrom.pop("target1_")} | soqs
    
    def _reflection_core(
        self, bb: BloqBuilder, i: int, ra: SoquetT, state: List[SoquetT]
    ) -> Tuple[SoquetT, List[SoquetT]]:
        ra, state = self._prepare_i_state(bb, i, ra, state)
        ra = bb.add(Hadamard(), q=ra)
        ra = bb.add(ZGate(), q=ra)
        ra = bb.add(XGate(), q=ra)
        all_reg = bb.join(np.array([ra, *state]))
        all_reg = self._reflect(bb, all_reg)
        qubits = bb.split(all_reg)
        ra = qubits[0]
        state = qubits[1:]
        ra = bb.add(XGate(), q=ra)
        ra = bb.add(ZGate(), q=ra)
        ra = bb.add(Hadamard(), q=ra)
        ra, state = self._prepare_i_state(bb, i, ra, state)
        return ra, state

    def _prepare_i_state(
        self, bb: BloqBuilder, i: int, refl_ancilla: SoquetT, state: List[SoquetT]
    ) -> Tuple[SoquetT, List[SoquetT]]:
        for i, bit in enumerate(f"{i:0{self.gate_bitsize}b}"):
            if bit == '1':
                refl_ancilla, state[i] = bb.add(CNOT(), ctrl=refl_ancilla, target=state[i])
        return refl_ancilla, state

    def _reflect(self, bb: BloqBuilder, reg: SoquetT):
        mult_control_flip = MultiControlPauli(
            cvs=tuple([0] * (self.gate_bitsize + 1)), target_gate=cirq.Z
        )
        ancilla = bb.allocate(1)
        ancilla = bb.add(XGate(), q=ancilla)
        reg, ancilla = bb.add(mult_control_flip, controls=reg, target=ancilla)
        ancilla = bb.add(XGate(), q=ancilla)
        bb.free(ancilla)
        return reg


@attrs.frozen
class DifferentialQROM(Bloq):
    prev_angles: Tuple[Tuple[float, ...], ...]
    new_angles: Tuple[Tuple[float, ...], ...]
    target_bitsize: int
    control_bitsize: int = 0

    def __attrs_post_init__(self):
        assert len(self.prev_angles) == len(self.new_angles), "different amount of angle register"
        assert len(set([len(a) for a in self.prev_angles])) == 1, "varying number of angles for prev"
        assert len(set([len(a) for a in self.new_angles])) == 1, "varying number of angles for post"
        assert self.target_bitsize > 0, "size of the target register must be greater than zero"

    @property
    def selection_bitsize(self) -> int:
        return (max(len(self.prev_angles[0]), len(self.new_angles[0])) - 1).bit_length()

    @property
    def target_soqs(self) -> Dict[str, int]:
        return dict([(f"target{i}_", self.target_bitsize) for i in range(len(self.prev_angles))])

    @property
    def signature(self) -> Signature:
        return Signature.build(
            control=self.control_bitsize, selection=self.selection_bitsize, **self.target_soqs
        )

    @property
    def rom_values(self) -> Tuple[Tuple[int, ...], ...]:
        rv = []
        for ap, an in zip(self.prev_angles, self.new_angles):
            rv_list = tuple(
                [
                    self._angle_to_rom_value(ap[i//int(2**self.selection_bitsize/len(ap))]) ^
                    self._angle_to_rom_value(an[i//int(2**self.selection_bitsize/len(an))])
                    for i in range(2**self.selection_bitsize)
                ]
            )
            rv.append(rv_list)
        return tuple(rv)

    def _angle_to_rom_value(self, angle: float) -> int:
        r"""Given an angle, returns the value to be loaded in ROM.

        Returns the value to be loaded to a QROM to encode the given angle with a certain value of
        phase_bitsize.
        """
        rom_value_decimal = 2**self.target_bitsize * angle / (2 * np.pi)
        return round(rom_value_decimal) % (2**self.target_bitsize)

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        print(f"loading: {self.rom_values}")
        soqs = bb.add_d(
            QROM(
                [np.array(rv) for rv in self.rom_values],
                selection_bitsizes=(self.selection_bitsize,),
                target_bitsizes=(self.target_bitsize,) * len(self.rom_values),
            ),
            **soqs,
        )
        return soqs


@attrs.frozen
class RotationViaAddition(Bloq):
    # [target reg, [control qubit applied to, pre bloq, post bloq]]
    pre_post_gates: Tuple[int, Tuple[int, Bloq, Bloq]]
    num_angle_soqs: int
    target_bitsize: int
    num_controls: int

    @property
    def angle_soqs(self) -> Dict[str, int]:
        return dict([(f"target{i}_", self.target_bitsize) for i in range(self.num_angle_soqs)])

    @property
    def signature(self) -> Signature:
        return Signature.build(
            control=self.num_controls, **self.angle_soqs, phase_grad=self.target_bitsize
        )
    
    @property
    def pre_post_all(self) -> Tuple[int]:
        indices = [ind for ind, _ in self.pre_post_gates]
        pre_post_all = []
        for i in range(self.num_angle_soqs):
            if i in indices:
                pre_post_all.append(self.pre_post_gates[indices.index(i)][1])
            else:
                pre_post_all.append((0, None, None))
        return tuple(pre_post_all)

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        print(self.pre_post_all)
        adder = AddIntoPhaseGrad(self.target_bitsize, self.target_bitsize)
        for i in range(self.num_angle_soqs):
            soqs = self.apply_ith_rotation(i, bb, adder, **soqs)
        return soqs

    def apply_ith_rotation(
        self,
        i: int,
        bb: BloqBuilder,
        adder: Bloq,
        **soqs: SoquetT,
    ):
        pre_post = self.pre_post_all[i]
        control = bb.split(soqs.pop("control"))
        if pre_post[1] is not None:
            control[pre_post[0]] = bb.add(pre_post[1], q=control[pre_post[0]])
        buffer = bb.allocate(self.target_bitsize)
        control[pre_post[0]], soqs[f"target{i}_"], buffer = bb.add(
            CSwapApprox(self.target_bitsize),
            ctrl=control[pre_post[0]],
            x=soqs[f"target{i}_"],
            y=buffer,
        )
        buffer, soqs["phase_grad"] = bb.add(adder, x=buffer, phase_grad=soqs["phase_grad"])
        control[pre_post[0]], soqs[f"target{i}_"], buffer = bb.add(
            CSwapApprox(self.target_bitsize),
            ctrl=control[pre_post[0]],
            x=soqs[f"target{i}_"],
            y=buffer,
        )
        if pre_post[2] is not None:
            control[pre_post[0]] = bb.add(pre_post[2], q=control[pre_post[0]])
        soqs["control"] = bb.join(control)
        bb.free(buffer)
        return soqs