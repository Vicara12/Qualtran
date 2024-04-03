from typing import Dict, List, Tuple

import attrs
import cirq
import numpy as np

from qualtran import Bloq, BloqBuilder, Signature, SoquetT
from qualtran.bloqs.basic_gates import CNOT, Hadamard, XGate, ZGate
from qualtran.bloqs.mcmt.multi_control_multi_target_pauli import MultiControlPauli
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState


@attrs.frozen
class SynthesizeGateHR(Bloq):

    cols: Tuple[Tuple[int, Tuple[complex, ...]], ...]
    phase_bitsize: int
    internal_phase_grad: bool = False

    @property
    def gate_bitsize(self):
        return (len(self.cols[0][1]) - 1).bit_length()

    @property
    def signature(self) -> Signature:
        return Signature.build(
            reflection_ancilla=1,
            state=self.gate_bitsize,
            phase_grad=self.phase_bitsize * (not self.internal_phase_grad),
        )

    def build_composite_bloq(
        self, bb: BloqBuilder, reflection_ancilla: SoquetT, state: SoquetT, **soqs: SoquetT
    ) -> Dict[str, SoquetT]:
        state_qubits = bb.split(state)
        reflection_ancilla, state_qubits = self._reflection_core(
            bb, 1, reflection_ancilla, state_qubits
        )
        state = bb.join(state_qubits)
        return {"reflection_ancilla": reflection_ancilla, "state": state} | soqs

    def _ith_reflection(self):
        pass

    def _r_ui_t(
        self, bb: BloqBuilder, i: int, ra: SoquetT, state: List[SoquetT]
    ) -> Tuple[SoquetT, List[SoquetT]]:
        return ra, state

    def _r_ui(
        self, bb: BloqBuilder, i: int, ra: SoquetT, state: List[SoquetT]
    ) -> Tuple[SoquetT, List[SoquetT]]:
        return ra, state

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
