from __future__ import annotations
import attrs
from typing import Tuple, Dict
from numpy.typing import ArrayLike

from quimb.tensor import MatrixProductState
import numpy as np
import scipy as scp

from qualtran import Bloq, Signature, BloqBuilder, SoquetT
from qualtran.bloqs.chemistry.prepare_mps.decompose_gate_hr import DecomposeGateViaHR
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState


@attrs.frozen
class PrepareMPS (Bloq):
    phase_bitsize: int
    tensors: Tuple[Tuple]
    # control_bitsize: int = 0
    uncompute: bool = False
    internal_phase_gradient: bool = True

    @property
    def signature(self):
        # return Signature.build(control=self.control_bitsize, input_state=self.state_bitsize, phase_gradient=self.phase_bitsize)
        return Signature.build(input_state=self.state_bitsize, phase_gradient=(not self.internal_phase_gradient)*self.phase_bitsize)
    
    @property
    def state_bitsize(self):
        return len(self.tensors)
    
    def build_composite_bloq(self, bb: BloqBuilder, *, input_state: SoquetT, **soqs: SoquetT) -> Dict[str, SoquetT]:
        """Extra soquets inside soqs are:
            * phase_grad: a phase gradient state of size phase_bitsize if internal_phase_gradient
                    is set to False
        """
        if self.internal_phase_gradient:
            phase_gradient = bb.add(PhaseGradientState(self.phase_bitsize))
        else:
            phase_gradient = soqs.pop("phase_grad")
        gates = self.gates_from_tensors()
        input_qubits = bb.split(input_state)
        for i in list(range(self.state_bitsize))[::(1-2*self.uncompute)]:
            gate_size = (len(gates[i])-1).bit_length()
            input_qs = bb.join(input_qubits[i:(i+gate_size)])
            ra = bb.allocate(1)
            gate_cols = tuple([(i, tuple(gc)) for i, gc in enumerate(gates[i].T)])
            gate_compiler = DecomposeGateViaHR(self.phase_bitsize, gate_cols, self.uncompute)
            input_qs, phase_gradient, ra = bb.add(gate_compiler, gate_input=input_qs, phase_grad=phase_gradient, reflection_ancilla=ra)
            bb.free(ra)
            input_qubits[i:(i+gate_size)] = bb.split(input_qs)
        input_state = bb.join(input_qubits)
        if self.internal_phase_gradient:
            bb.add(PhaseGradientState(self.phase_bitsize).adjoint(), phase_grad=phase_gradient)
        else:
            soqs["phase_grad"] = phase_gradient
        return {"input_state": input_state} | soqs
    
    @staticmethod
    def fill_gate (gate):
        ker = scp.linalg.null_space(gate.T)
        return np.hstack((gate, ker))

    @staticmethod
    def revert_dims (M, dims):
        for d in dims:
            shape = M.shape
            wires = (shape[d]-1).bit_length()
            divided = shape[:d] + (2,)*wires + shape[d+1:]
            reorder = tuple(range(d)) + tuple(range(d+wires-1,d-1,-1)) + tuple(range(d+wires,len(shape)+wires-1))
            M = M.reshape(divided).transpose(reorder).reshape(shape)
        return M

    def gates_from_tensors (self):
        bitsize = len(self.tensors)
        gates = []
        if len(self.tensors) > 1:
            gates.append(PrepareMPS.fill_gate(np.array(self.tensors[0]).T.reshape((-1,1))))
        for i in range(1,bitsize-1):
            tensor = PrepareMPS.revert_dims(np.array(self.tensors[i]),[1]).T
            gates.append(PrepareMPS.revert_dims(self.fill_gate(tensor.reshape((-1,tensor.shape[2]))),[1]))
        gates.append(PrepareMPS.fill_gate(np.array(self.tensors[-1]).T.reshape((2,-1))))
        return gates
    
    @staticmethod
    def from_quimb_mps (mps: MatrixProductState, phase_bitsize: int, uncompute: bool = False) -> PrepareMPS:
        tensors = [t.data for t in mps]
        tensors[0] = tuple([tuple(l) for l in tensors[0]])
        for i in range(1,len(tensors)-1):
            tensors[i] = tuple([tuple([tuple(i) for i in l]) for l in tensors[i]])
        tensors[-1] = tuple([tuple(l) for l in tensors[-1]])
        return PrepareMPS(tensors=tuple(tensors), phase_bitsize=phase_bitsize, uncompute=uncompute)
