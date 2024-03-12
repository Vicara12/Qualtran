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

from typing import Tuple

import numpy as np
import pytest

from qualtran import BloqBuilder
from quimb.tensor.tensor_1d import MatrixProductState
from qualtran.bloqs.chemistry.prepare_mps.prepare_mps import PrepareMPS
from qualtran.bloqs.state_preparation.state_preparation_via_rotation import StatePreparationViaRotations
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from qualtran.testing import assert_valid_bloq_decomposition


@pytest.mark.parametrize(
    "phase_bitsize, state",
    [
        # 1 site
        [3, ((-0.13483551719393483-0.6792138311871981j), (0.27792548262621136+0.6657667616621002j))],
        # 2 sites
        [3, ((0.34516117955291464-0.47050857553445424j), (-0.44495179644249855+0.333870947031753j), (0.076814114542036+0.350561936676884j), (-0.032172546251369775-0.4692593927425125j))],
        # 4 sites
        [4, ((-0.07682732490283989+0.15147492902068269j), (-0.013084036514757464-0.22425890159592426j), (0.08118165220365978-0.05522733648035143j), (-0.11533741257627657-0.035389807420842134j), (0.0002812925262672198+0.023951209732558983j), (-0.10168718952348461-0.3029116191292497j), (0.2985542757093312-0.13775558345124608j), (0.11577702666216419+0.004595530607986605j), (0.2428967980069414-0.14829932701377663j), (-0.1942374440925404-0.02811171570596936j), (-0.19387329968261618-0.2392196426426074j), (0.2566854641869052-0.20083381626523344j), (0.06594095898948661+0.035952610911029415j), (0.10081979402101338+0.29624055051276427j), (-0.02502095387266981+0.3018650181057659j), (0.2930486993328657+0.2656187168382382j))],
    ],
)
def test_untrimmed_mps(phase_bitsize: int, state: Tuple[complex, ...]):
    sites = (len(state)-1).bit_length()
    mps = MatrixProductState.from_dense(state, dims=(2,)*sites)
    mps_prep = PrepareMPS.from_quimb_mps(mps, phase_bitsize)
    assert_valid_bloq_decomposition(mps_prep)
    bb = BloqBuilder()
    state = bb.allocate(sites)
    state = bb.add(mps_prep, input_state=state)
    coefs = bb.finalize(state=state).tensor_contract()
    assert abs(np.dot(mps.to_dense().conj()[:,0], coefs)) > 0.90


@pytest.mark.parametrize(
    "phase_bitsize, state",
    [
        # 1 site
        [4, ((-0.04413935515783222+0.6209151873131493j), (0.7748210001862302+0.11031076629475997j))],
        # 2 sites
        [4, ((0.34516117955291464-0.47050857553445424j), (-0.44495179644249855+0.333870947031753j), (0.076814114542036+0.350561936676884j), (-0.032172546251369775-0.4692593927425125j))],
    ],
)
def test_prepare_mps_adjoint(phase_bitsize: int, state: Tuple[complex, ...]):
    sites = (len(state)-1).bit_length()
    mps = MatrixProductState.from_dense(state, dims=(2,)*sites)
    mps_prep_adjoint = PrepareMPS.from_quimb_mps(mps, phase_bitsize, uncompute=True, internal_phase_gradient=False)
    state_prep = StatePreparationViaRotations(phase_bitsize, state)
    assert_valid_bloq_decomposition(mps_prep_adjoint)
    bb = BloqBuilder()
    state = bb.allocate(sites)
    pg = bb.add(PhaseGradientState(phase_bitsize))
    state, pg = bb.add(state_prep, target_state=state, phase_gradient=pg)
    state, pg = bb.add(mps_prep_adjoint, input_state=state, phase_grad=pg)
    bb.add(PhaseGradientState(phase_bitsize).adjoint(), phase_grad=pg)
    coefs = bb.finalize(state=state).tensor_contract()
    assert abs(coefs[0]) > 0.96