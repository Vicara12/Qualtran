import attrs
from typing import Tuple, Dict
from typing_extensions import Self
from numpy.typing import ArrayLike

from quimb.tensor import MatrixProductState
import numpy as np
from numpy.typing import ArrayLike

from qualtran import Bloq, Signature, BloqBuilder, SoquetT, bloq_example, BloqDocSpec
from qualtran.bloqs.chemistry.prepare_mps.decompose_gate_hr import DecomposeGateViaHR
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState


@attrs.frozen
class PrepareMPS (Bloq):
    r"""Prepares a quantum state encoded in an MPS.

    Given the tensors that encode a MPS or a Quimb MatrixProductState object, this bloq prepares
    the equivalent quantum state using the method from [1].

    A phase gradient state is needed for this process. It can be provided externally to improve the
    T-cost of the algorithm (specially if the process is repeated multiple times), otherwise it is
    generated and deallocated internally.

    This Bloq can be built directly with the constructor, in which case the tensors that form the
    MPS must be provided, or from a Quimb MatrixProductState, in which case it is necessary to use
    the from_quimb_mps method.

    Args:
        tensors: tuple of tensors (in tuple format) that encode the MPS in left canonical form.
            The format in the case of a single site MPS is `(((coef_0, coef_1),),)`. For a two site
            mps the tensor disposition is
            `([bond_dim_0, physical_ind_0], [bond_dim_0, physical_ind_1])` for the first and second
            sites. In a n-site MPS the disposition is `([bond_dim_0, physical_ind_0], ...,
            [bond_dim_{i-1}, bond_dim_i, physical_ind_i], ...,
            [bond_dim_{n-2}, physical_ind_{n-1}])`. For an example of this encoding refer to the
            tutorial.
        phase_bitsize: size of the register that is used to store the rotation angles when loading
            the tensor values. Bigger values increase the accuracy of the results but require
            approximately 2 extra ancilla per qubit of phase_bitsize.
        uncompute: wether to implement the MPS preparation circuit or its adjoint.
        internal_phase_gradient: a phase gradient state is needed for the decomposition. It can be
            either be provided externally if this attribute is set to False or internally otherwise.

    References:
        [Sequential Generation of Entangled Multiqubit States]
        (https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.95.110503).
            C. Schön, E. Solano, F. Verstraete, J. I. Cirac, and M. M. Wolf. 2005.
    """
    tensors: Tuple[Tuple]
    phase_bitsize: int
    uncompute: bool = False
    internal_phase_gradient: bool = True

    @property
    def signature(self) -> Signature:
        # return Signature.build(control=self.control_bitsize, input_state=self.state_bitsize, phase_gradient=self.phase_bitsize)
        return Signature.build(input_state=self.state_bitsize, phase_grad=(not self.internal_phase_gradient)*self.phase_bitsize)
    
    @property
    def state_bitsize(self) -> int:
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
        gates = self._gates_from_tensors()
        input_qubits = bb.split(input_state)
        for i in list(range(self.state_bitsize))[::(1-2*self.uncompute)]:
            gate_size = (len(gates[i][0][1])-1).bit_length()
            input_qs = bb.join(input_qubits[i:(i+gate_size)])
            gate_compiler = DecomposeGateViaHR(self.phase_bitsize, gates[i], self.uncompute)
            input_qs, phase_gradient = bb.add(gate_compiler, gate_input=input_qs, phase_grad=phase_gradient)
            input_qubits[i:(i+gate_size)] = bb.split(input_qs)
        input_state = bb.join(input_qubits)
        if self.internal_phase_gradient:
            bb.add(PhaseGradientState(self.phase_bitsize).adjoint(), phase_grad=phase_gradient)
        else:
            soqs["phase_grad"] = phase_gradient
        return {"input_state": input_state} | soqs

    @staticmethod
    def _revert_dims (M: ArrayLike, dims: Tuple[int,...]):
        r""" Transposes a single dimension of a tensor.
        For example, in a tensor of shape (4,8,16) reverting the dimension 1 is equivalent to first
        splitting the dimension 1 into subspaces of size 2, which results in a new shape of
        (4,2,2,2,16), which for illustration will be labeled as (4,2a,2b,2c,16). Then a transpose
        of the subspaces that formed the dimension to be reverted is performed, which results in
        (4,2c,2b,2a,16), and finally a rejoin that leaves the tensor again in the shape (4,8,16).
        """
        for d in dims:
            shape = M.shape
            wires = (shape[d]-1).bit_length()
            divided = shape[:d] + (2,)*wires + shape[d+1:]
            reorder = tuple(range(d)) + tuple(range(d+wires-1,d-1,-1)) + tuple(range(d+wires,len(shape)+wires-1))
            M = M.reshape(divided).transpose(reorder).reshape(shape)
        return M

    def _gates_from_tensors (self) -> Tuple[ArrayLike,...]:
        bitsize = len(self.tensors)
        gates = []
        # if the MPS is a single site long then the only gate is the last one
        if len(self.tensors) > 1:
            gates.append(((0, tuple(np.array(self.tensors[0]).T.reshape((-1)))),))
        for i in range(1,bitsize-1):
            # the revert dims and rev_i are used to reverse the indices of the bond dimension
            # used by Quimb (as the convention used by them is the opposite), this means that
            # the item at bond dimension index 0101 gets transposed to position 1010
            tensor = PrepareMPS._revert_dims(np.array(self.tensors[i]),[1]).T
            gate_cols_data = tensor.reshape((-1,tensor.shape[2])).T
            blen = (gate_cols_data[0].shape[0] - 1).bit_length()
            def rev_i (index: int, blen: int) -> int: return int(f"{index:0{blen}b}"[::-1], 2)
            gate_cols = [(rev_i(i, blen), tuple(gcd)) for i, gcd in enumerate(gate_cols_data)]
            gates.append(tuple(gate_cols))
        last_gate_cols = np.array(self.tensors[-1]).T.reshape((2,-1)).T
        gates.append(tuple([(i, tuple(gcd)) for i, gcd in enumerate(last_gate_cols)]))
        return gates
    
    @staticmethod
    def tensor_to_tuple (T: ArrayLike) -> Tuple:
        if len(T.shape) == 1:
            return tuple(T)
        return tuple(map(PrepareMPS.tensor_to_tuple, T))

    @staticmethod
    def _extract_tensors (mps: MatrixProductState) -> Tuple[ArrayLike,...]:
        r""" Extracts the tensors with the desired index order.
        Sometimes Quimb might reorder internal indices, the correct order used in this bloq is:
          (bond_0, physical_0) for the first site
          (bond_{i-1}, bond_i, physical_i) for the internal sites
          (bond_{n-2}, physical_{n-1}) for the last site
        """
        # compute the index reordering (transposition) necessary to set the tensor in the correct
        # format
        virt_inds = mps.inner_inds()
        phys_inds = mps.outer_inds()
        sites = len(phys_inds)
        # if the MPS is one site long then there is just one index, no reordering needed
        if sites == 1:
            return [PrepareMPS.tensor_to_tuple(mps[0].data)]
        corr_inds = [(virt_inds[0], phys_inds[0])] +\
                    [(virt_inds[i-1], virt_inds[i], phys_inds[i]) for i in range(1,sites-1)] +\
                    [(virt_inds[sites-2], phys_inds[sites-1])]
        transpositions = []
        for i in range(sites):
            transpositions.append([mps[i].inds.index(ind) for ind in corr_inds[i]])
        # for each site, get its coefficient tensor reordered according to what was computed before
        reordered = [np.transpose(site.data, transp) for site, transp in zip(mps, transpositions)]
        return [PrepareMPS.tensor_to_tuple(t) for t in reordered]

    
    @staticmethod
    def from_quimb_mps (mps: MatrixProductState, phase_bitsize: int, **kwargs) -> Self:
        r"""Constructs a MPS preparation bloq from a Quimb MPS object.
        Arguments are a Quimb MatrixProductState object and all the others that the default
        constructor of PrepareMPS receives, except for tensors.
        The bond dimensions of the mps must be powers of two.
        """
        mps.compress()
        tensors = PrepareMPS._extract_tensors(mps)
        return PrepareMPS(tensors=tuple(tensors), phase_bitsize=phase_bitsize, **kwargs)


@bloq_example
def _prepare_mps() -> PrepareMPS:
    tensors = (
        (((-0.6221018876629202+0.6420514495011711j), (0.132495199023149+0.3193914665424655j)),
         ((0.10238069190497784-0.009166000913358544j), (-0.0832597254195061+0.2523792530512902j))),

        (((-0.11429513645729317+0j), (-0.8147485065475832+0.5684377651605244j)),
         ((-0.9934468389311071+0j), (0.09373605922831829-0.06539823711795652j)))
         )
    prepare_mps = PrepareMPS(tensors=tensors, phase_bitsize=3)
    return prepare_mps


_MPS_PREPARATION_DOC = BloqDocSpec(
    bloq_cls=PrepareMPS,
    import_line='from qualtran.bloqs.chemistry.prepare_mps.prepare_mps import PrepareMPS',
    examples=(_prepare_mps,),
)