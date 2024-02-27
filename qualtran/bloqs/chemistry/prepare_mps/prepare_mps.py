import attrs
from typing import Tuple, Dict
from numpy.typing import ArrayLike

from qualtran import Bloq, Signature, BloqBuilder, SoquetT
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.bloqs.chemistry.prepare_mps.compile_gate import CompileGateGivenVectors


@attrs.frozen
class PrepareMPS (Bloq):
    gate_tensors: Tuple[ArrayLike]
    uncompute: bool = False

    @property
    def signature(self):
        return Signature.build(input_state=self.state_bitsize)
    
    @property
    def state_bitsize(self):
        return len(self.gate_tensors)
    
    def build_composite_bloq(self, bb: BloqBuilder, *, input_state: SoquetT) -> Dict[str, SoquetT]:
        return {"input_state": input_state}