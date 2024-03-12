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
from qualtran.testing import assert_valid_bloq_decomposition


# these gates can be approximated exactly with the given phase_bitsize
@pytest.mark.parametrize(
    "phase_bitsize, state",
    [
        [5, ((-0.38810603431711177-0.07583019068904245j), (0.19139378922915132+0.4564481162899791j), (-0.4017275139496865-0.05808888166710542j), (-0.01887666337596189+0.300269601577422j), (0.16026173942665436+0.12265359746616425j), (-0.24729396696973335+0.3085668229904108j), (0.18469527788499457+0.2581220559684759j), (-0.17245207825615652+0.12567802515393806j))],
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
    assert abs(np.dot(mps.to_dense().conj()[:,0], coefs)) > 0.95