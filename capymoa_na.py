# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CapyMOA wrapper for the refactored NA (GroCH / Growing Concept History) classifier.

Usage
-----
Point ``CAPYMOA_MOA_JAR`` to a MOA 2024.07 JAR containing the compiled ``NA``,
``GNG``, ``ADDM``, ``DynamicWindowClassificationPerformanceEvaluator``, and
distance classes **before** importing ``capymoa``::

    export CAPYMOA_MOA_JAR=/path/to/moa_with_na.jar

    from capymoa.stream import ARFFStream
    from capymoa.evaluation import prequential_evaluation
    from capymoa_na import NA

    stream = ARFFStream(path="data.arff")
    learner = NA(schema=stream.get_schema(), CLI="-l (trees.HoeffdingTree) ...")
    results = prequential_evaluation(stream=stream, learner=learner, max_instances=50000)
"""

from __future__ import annotations

from capymoa.base import MOAClassifier
from capymoa.stream import Schema
from moa.classifiers.meta import NA as _MOA_NA


class NA(MOAClassifier):
    """CapyMOA ``MOAClassifier`` wrapper around ``moa.classifiers.meta.NA``."""

    def __init__(
        self,
        schema: Schema,
        CLI: str | None = None,
        random_seed: int = 1,
    ):
        super().__init__(
            schema=schema,
            CLI=CLI,
            random_seed=random_seed,
            moa_learner=_MOA_NA,
        )
