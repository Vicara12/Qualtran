from __future__ import annotations
import attrs
from typing import Tuple, Dict
from numpy.typing import ArrayLike

from quimb.tensor import MatrixProductState
import numpy as np
import scipy as scp

from qualtran import Bloq, Signature, BloqBuilder, SoquetT
from qualtran.bloqs.chemistry.prepare_mps.compile_gate import CompileGateGivenVectorsWithoutPG


@attrs.frozen
class PrepareMPS (Bloq):
    phase_bitsize: int
    tensors: Tuple[ArrayLike]
    uncompute: bool = False

    @property
    def signature(self):
        return Signature.build(input_state=self.state_bitsize)
    
    @property
    def state_bitsize(self):
        return len(self.tensors)
    
    def build_composite_bloq(self, bb: BloqBuilder, *, input_state: SoquetT) -> Dict[str, SoquetT]:
        gate = self.gates_from_tensors()
        input_qubits = bb.split(input_state)
        for i in range(self.state_bitsize):
            if self.uncompute:
                qi = self.state_bitsize - i - 1
            else:
                qi = i
            gate_size = (len(gate[qi])-1).bit_length()
            input_qs = bb.join(input_qubits[qi:(qi+gate_size)])
            gate_compiler = CompileGateGivenVectorsWithoutPG(self.phase_bitsize, gate[i].T, self.uncompute)
            input_qs = bb.add(gate_compiler, gate_input=input_qs)
            input_qubits[qi:(qi+gate_size)] = bb.split(input_qs)
        return {"input_state": input_state}
    
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
            gates.append(PrepareMPS.fill_gate(self.tensors[0].T.reshape((-1,1))))
        for i in range(1,bitsize-1):
            tensor = PrepareMPS.revert_dims(self.tensors[i],[1]).T
            gates.append(PrepareMPS.revert_dims(self.fill_gate(tensor.reshape((-1,tensor.shape[2]))),[1]))
        gates.append(PrepareMPS.fill_gate(self.tensors[-1].T.reshape((2,-1))))
        return gates
    
    @staticmethod
    def from_quimb_mps (mps: MatrixProductState, phase_bitsize: int, uncompute: bool = False) -> PrepareMPS:
        tensors = [t.data for t in mps]
        return PrepareMPS(tensors=tensors, phase_bitsize=phase_bitsize, uncompute=uncompute)
