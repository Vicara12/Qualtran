from typing import Tuple, Dict

from qualtran import Bloq, Signature, BloqBuilder, SoquetT


class SynthesizeGateHR (Bloq):

    cols: Tuple[Tuple[int, Tuple[complex,...]],...]
    phase_bitsize: int

    @property
    def bitsize (self):
        return (len(self.cols[0][1])-1).bit_length()

    @property
    def signature(self) -> Signature:
        return Signature.build(reflection_ancilla=1, state=self.bitsize, phase_grad=self.phase_bitsize)
    
    def build_composite_bloq(self, bb: BloqBuilder, reflection_ancilla: SoquetT, state: SoquetT, phase_grad: SoquetT) -> Dict[str, SoquetT]:
        pass