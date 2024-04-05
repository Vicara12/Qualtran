from typing import Dict, List, Tuple

import attrs
import cirq
import numpy as np

from qualtran import Bloq, BloqBuilder, Signature, SoquetT
from qualtran.bloqs.basic_gates import CNOT, Hadamard, XGate, ZGate
from qualtran.bloqs.basic_gates.rotation import Rx
from qualtran.bloqs.basic_gates.swap import CSwap
from qualtran.bloqs.mcmt.and_bloq import And
from qualtran.bloqs.mcmt.multi_control_multi_target_pauli import MultiControlPauli
from qualtran.bloqs.qrom import QROM
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad, PhaseGradientState
from qualtran.bloqs.state_preparation.state_preparation_via_rotation import RotationTree
from qualtran.drawing import show_bloq


@attrs.frozen
class SynthesizeGateViaHR(Bloq):

    cols: Tuple[Tuple[int, Tuple[complex, ...]], ...]
    phase_bitsize: int
    internal_phase_grad: bool = False
    internal_refl_ancilla: bool = True
    optimize: bool = True

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

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        if self.internal_refl_ancilla:
            soqs["ra"] = bb.allocate(1)
            soqs["ra"] = bb.add(XGate(), q=soqs["ra"])
        else:
            soqs["ra"] = soqs.pop("reflection_ancilla")
        if self.internal_phase_grad:
            soqs["phase_grad"] = bb.add(PhaseGradientState(self.phase_bitsize))
        soqs["ph_reg"] = bb.allocate(self.phase_bitsize)
        soqs["amp_reg"] = bb.allocate(self.phase_bitsize)
        amps_pre = (0,)
        amps_left = (0,)
        phases_pre = (0,)
        phases_left = (0,)
        for index, (i, ui) in enumerate(self.cols):
            amps, phases, gl = RotationTree(ui, False).get_angles()
            amps_t, phases_t, gl_t = RotationTree(ui, True).get_angles()
            if index != 0 and self.optimize:
                if self.gate_bitsize > 1:
                    soqs = self._preparation_transition(
                        bb,
                        amps_pre,
                        phases_pre,
                        amps_left,
                        phases_left,
                        phases_t[-1],
                        amps_t[-1],
                        phases_t[-2],
                        amps_t[-2],
                        **soqs,
                    )
                else:
                    soqs = self._preparation_transition(
                        bb,
                        amps_pre,
                        phases_pre,
                        amps_left,
                        phases_left,
                        phases_t[-1],
                        amps_t[-1],
                        (0,),
                        (0,),
                        **soqs,
                    )
            soqs = self._r_ui_t(bb, amps_t, phases_t, gl_t, index == 0 or not self.optimize, **soqs)
            soqs["ra"], soqs["state"] = self._reflection_core(bb, i, soqs["ra"], soqs["state"])
            soqs = self._r_ui(bb, amps, phases, gl, index == len(self.cols) - 1 or not self.optimize, **soqs)
            if self.gate_bitsize > 1:
                amps_pre = amps[-2]
                phases_pre = phases[-2]
            amps_left = amps[-1]
            phases_left = phases[-1]
        bb.free(soqs.pop("ph_reg"))
        bb.free(soqs.pop("amp_reg"))
        if self.internal_phase_grad:
            bb.add(
                PhaseGradientState(self.phase_bitsize).adjoint(), phase_grad=soqs.pop("phase_grad")
            )
        if self.internal_refl_ancilla:
            bb.free(soqs.pop("ra"))
        else:
            soqs["reflection_ancilla"] = soqs.pop("ra")
        return soqs

    def _r_ui_t(
        self,
        bb: BloqBuilder,
        amps_t: Tuple[Tuple[float, ...], ...],
        phases_t: Tuple[Tuple[float, ...], ...],
        global_ph: float,
        complete: bool,
        **soqs: SoquetT,
    ) -> Dict[str, SoquetT]:
        r"""Registers are:
        * ra: reflection ancilla
        * state: input qubits
        * amp_reg: register to store the amplitude angle
        * ph_reg: register to store the phase angle
        * phase_grad: phase gradient register
        """
        pre_post_gates = ((1, (-1, Rx(angle=np.pi / 2), Rx(angle=-np.pi / 2))),)
        qubits = bb.split(soqs.pop("state"))
        soqs_qrom = {"target0_": soqs.pop("ph_reg"), "target1_": soqs.pop("amp_reg")}
        if complete:
            soqs["ra"] = bb.add(XGate(), q=soqs["ra"])
            if self.gate_bitsize > 1:
                soqs_qrom["selection"] = bb.join(qubits[: (self.gate_bitsize - 1)])
            qrom = DifferentialQROM(((0,), (0,)), (phases_t[-1], amps_t[-1]), self.phase_bitsize)
            soqs_qrom = bb.add_d(qrom, **soqs_qrom)
            if self.gate_bitsize > 1:
                qubits[: (self.gate_bitsize - 1)] = bb.split(soqs_qrom.pop("selection"))
        if complete:
            iter_values = list(enumerate(zip(amps_t, phases_t)))[::-1]
        else:
            # remove first entry
            iter_values = list(enumerate(zip(amps_t, phases_t)))[-2::-1]
        for i, (amp_qubit, ph_qubit) in iter_values:
            adders = RotationViaAddition(pre_post_gates, 2, self.phase_bitsize, 2)
            control = bb.join(np.array([soqs.pop("ra"), qubits[i]]))
            control, soqs_qrom["target0_"], soqs_qrom["target1_"], soqs["phase_grad"] = bb.add(
                adders,
                control=control,
                target0_=soqs_qrom["target0_"],
                target1_=soqs_qrom["target1_"],
                phase_grad=soqs["phase_grad"],
            )
            control_qubits = bb.split(control)
            soqs["ra"] = control_qubits[0]
            qubits[i] = control_qubits[1]
            if i != 0:
                angles = (phases_t[i - 1], amps_t[i - 1])
            else:
                angles = ((0,), (0,))
            qrom = DifferentialQROM((ph_qubit, amp_qubit), angles, self.phase_bitsize)
            if i != 0:
                soqs_qrom["selection"] = bb.join(qubits[:i])
            soqs_qrom = bb.add_d(qrom, **soqs_qrom)
            if i != 0:
                qubits[:i] = bb.split(soqs_qrom.pop("selection"))
        soqs["ph_reg"] = soqs_qrom.pop("target0_")
        soqs["amp_reg"] = soqs_qrom.pop("target1_")
        soqs["state"] = bb.join(qubits)
        soqs = self._state_global_phase(bb, global_ph, **soqs)
        soqs["ra"] = bb.add(XGate(), q=soqs["ra"])
        return soqs

    def _r_ui(
        self,
        bb: BloqBuilder,
        amps: Tuple[Tuple[float, ...], ...],
        phases: Tuple[Tuple[float, ...], ...],
        global_ph: float,
        complete: bool,
        **soqs: SoquetT,
    ) -> Dict[str, SoquetT]:
        r"""Registers are:
        * ra: reflection ancilla
        * state: input qubits
        * amp_reg: register to store the amplitude angle
        * ph_reg: register to store the phase angle
        * phase_grad: phase gradient register
        """
        soqs["ra"] = bb.add(XGate(), q=soqs["ra"])
        pre_post_gates = ((0, (-1, Rx(angle=np.pi / 2), Rx(angle=-np.pi / 2))),)
        qubits = bb.split(soqs.pop("state"))
        soqs = self._state_global_phase(bb, global_ph, **soqs)
        soqs_qrom = {"target0_": soqs.pop("amp_reg"), "target1_": soqs.pop("ph_reg")}
        if complete:
            iter_values = list(enumerate(zip(amps, phases)))
        else:
            iter_values = list(enumerate(zip(amps, phases)))[:-1]
        for i, (amp_qubit, ph_qubit) in iter_values:
            if i == 0:
                prev_angs = ((0,), (0,))
            else:
                prev_angs = (amps[i - 1], phases[i - 1])
            qrom = DifferentialQROM(prev_angs, (amp_qubit, ph_qubit), self.phase_bitsize)
            adders = RotationViaAddition(pre_post_gates, 2, self.phase_bitsize, 2)
            if i != 0:
                soqs_qrom["selection"] = bb.join(qubits[:i])
            soqs_qrom = bb.add_d(qrom, **soqs_qrom)
            if i != 0:
                qubits[:i] = bb.split(soqs_qrom.pop("selection"))
            control = bb.join(np.array([soqs.pop("ra"), qubits[i]]))
            control, soqs_qrom["target0_"], soqs_qrom["target1_"], soqs["phase_grad"] = bb.add(
                adders,
                control=control,
                target0_=soqs_qrom["target0_"],
                target1_=soqs_qrom["target1_"],
                phase_grad=soqs["phase_grad"],
            )
            control_qubits = bb.split(control)
            soqs["ra"] = control_qubits[0]
            qubits[i] = control_qubits[1]
        if complete:
            qrom = DifferentialQROM((amps[-1], phases[-1]), ((0,), (0,)), self.phase_bitsize)
            if i != 0:
                soqs_qrom["selection"] = bb.join(qubits[:-1])
            soqs_qrom = bb.add_d(qrom, **soqs_qrom)
            if i != 0:
                qubits[:-1] = bb.split(soqs_qrom.pop("selection"))
            soqs["ra"] = bb.add(XGate(), q=soqs["ra"])
        return {
            "state": bb.join(qubits),
            "amp_reg": soqs_qrom.pop("target0_"),
            "ph_reg": soqs_qrom.pop("target1_"),
        } | soqs

    def _preparation_transition(
        self,
        bb: BloqBuilder,
        amps_pre: List[float],
        phases_pre: List[float],
        amps_left: list[float],
        phases_left: list[float],
        phases_right: list[float],
        amps_right: List[float],
        phases_post: List[float],
        amps_post: List[float],
        **soqs: SoquetT,
    ):
        zero_angles = (0,) * len(amps_pre)
        phases_center = tuple([pl + pr for pl, pr in zip(phases_left, phases_right)])
        qubits = bb.split(soqs.pop("state"))
        soqs_qrom = {
            "target0_": soqs.pop("amp_reg"),
            "target1_": soqs.pop("ph_reg"),
            "target2_": bb.allocate(self.phase_bitsize),
        }
        control = bb.join(np.array([soqs.pop("ra"), qubits[-1]]))
        if self.gate_bitsize > 1:
            soqs_qrom["selection"] = bb.join(qubits[:-1])
        pre_post_gates = (
            (0, (-1, Rx(angle=np.pi / 2), Rx(angle=-np.pi / 2))),
            (2, (-1, Rx(angle=np.pi / 2), Rx(angle=-np.pi / 2))),
        )
        qrom_left = DifferentialQROM(
            (amps_pre, phases_pre, zero_angles),
            (amps_left, phases_center, amps_right),
            self.phase_bitsize,
        )
        qrom_right = DifferentialQROM(
            (amps_left, phases_center, amps_right),
            (amps_post, phases_post, zero_angles),
            self.phase_bitsize,
        )
        adders = RotationViaAddition(pre_post_gates, 3, self.phase_bitsize, 2)
        soqs_qrom = bb.add_d(qrom_left, **soqs_qrom)
        (
            control,
            soqs_qrom["target0_"],
            soqs_qrom["target1_"],
            soqs_qrom["target2_"],
            soqs["phase_grad"],
        ) = bb.add(
            adders,
            control=control,
            target0_=soqs_qrom["target0_"],
            target1_=soqs_qrom["target1_"],
            target2_=soqs_qrom["target2_"],
            phase_grad=soqs["phase_grad"],
        )
        soqs_qrom = bb.add_d(qrom_right, **soqs_qrom)
        if self.gate_bitsize > 1:
            qubits[:-1] = bb.split(soqs_qrom.pop("selection"))
        control_qubits = bb.split(control)
        qubits[-1] = control_qubits[1]
        bb.free(soqs_qrom.pop("target2_"))
        return {
            "ra": control_qubits[0],
            "state": bb.join(qubits),
            "amp_reg": soqs_qrom.pop("target0_"),
            "ph_reg": soqs_qrom.pop("target1_"),
        } | soqs

    def _state_global_phase(
        self, bb: BloqBuilder, global_phase: int, **soqs: SoquetT
    ) -> Dict[str, SoquetT]:
        rv = DifferentialQROM.angle_to_rom_value(self.phase_bitsize, global_phase)
        qrom = QROM(
            [np.array([rv])], selection_bitsizes=(0,), target_bitsizes=(self.phase_bitsize,)
        )
        adder = RotationViaAddition((), 1, self.phase_bitsize, 1)
        soqs["ph_reg"] = bb.add(qrom, target0_=soqs["ph_reg"])
        soqs["ra"], soqs["ph_reg"], soqs["phase_grad"] = bb.add(
            adder, control=soqs["ra"], target0_=soqs["ph_reg"], phase_grad=soqs["phase_grad"]
        )
        soqs["ph_reg"] = bb.add(qrom, target0_=soqs["ph_reg"])
        return soqs

    def _reflection_core(
        self, bb: BloqBuilder, i: int, ra: SoquetT, state: SoquetT
    ) -> Tuple[SoquetT, List[SoquetT]]:
        state = bb.split(state)
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
        state = bb.join(state)
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
        assert (
            len(set([len(a) for a in self.prev_angles])) == 1
        ), "varying number of angles for prev"
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
                    self._angle_to_rom_value(ap[i // int(2**self.selection_bitsize / len(ap))])
                    ^ self._angle_to_rom_value(an[i // int(2**self.selection_bitsize / len(an))])
                    for i in range(2**self.selection_bitsize)
                ]
            )
            rv.append(rv_list)
        return tuple(rv)

    def _angle_to_rom_value(self, angle: float) -> int:
        return DifferentialQROM.angle_to_rom_value(self.target_bitsize, angle)

    def angle_to_rom_value(phase_bitsize, angle: float) -> int:
        r"""Given an angle, returns the value to be loaded in ROM.

        Returns the value to be loaded to a QROM to encode the given angle with a certain value of
        phase_bitsize.
        """
        rom_value_decimal = 2**phase_bitsize * angle / (2 * np.pi)
        return round(rom_value_decimal) % (2**phase_bitsize)

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
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
        adder = AddIntoPhaseGrad(self.target_bitsize, self.target_bitsize)
        for i in range(self.num_angle_soqs):
            soqs = self.apply_ith_rotation(i, bb, adder, **soqs)
        return soqs

    def apply_ith_rotation(self, i: int, bb: BloqBuilder, adder: Bloq, **soqs: SoquetT):
        pre_post = self.pre_post_all[i]
        control = bb.split(soqs.pop("control"))
        if pre_post[1] is not None:
            control[pre_post[0]] = bb.add(pre_post[1], q=control[pre_post[0]])
        buffer = bb.allocate(self.target_bitsize)
        control = bb.join(control)
        control, soqs[f"target{i}_"], buffer = self.controlled_swap(
            bb, control, soqs[f"target{i}_"], buffer
        )
        buffer, soqs["phase_grad"] = bb.add(adder, x=buffer, phase_grad=soqs["phase_grad"])
        control, soqs[f"target{i}_"], buffer = self.controlled_swap(
            bb, control, soqs[f"target{i}_"], buffer
        )
        control = bb.split(control)
        if pre_post[2] is not None:
            control[pre_post[0]] = bb.add(pre_post[2], q=control[pre_post[0]])
        soqs["control"] = bb.join(control)
        bb.free(buffer)
        return soqs

    def controlled_swap(
        self, bb: BloqBuilder, ctrl: SoquetT, a: SoquetT, b: SoquetT
    ) -> Tuple[SoquetT, SoquetT, SoquetT]:
        if self.num_controls == 1:
            ctrl, a, b = bb.add(CSwap(self.target_bitsize), ctrl=ctrl, x=a, y=b)
        elif self.num_controls == 2:
            ctrl = bb.split(ctrl)
            ctrl, ctrl_ = bb.add(And(), ctrl=ctrl)
            ctrl_, a, b = bb.add(CSwap(self.target_bitsize), ctrl=ctrl_, x=a, y=b)
            ctrl = bb.add(And().adjoint(), ctrl=ctrl, target=ctrl_)
            ctrl = bb.join(ctrl)
        else:
            raise Exception("not implemented")
        return ctrl, a, b
