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

"""Portable smoke test for the CapyMOA NA wrapper (no hardcoded dataset paths).

Requires ``capymoa`` installed and ``CAPYMOA_MOA_JAR`` pointing to a MOA 2024.07
JAR containing the compiled ``NA`` and ``GNG`` classes. Skips cleanly otherwise.
"""

from __future__ import annotations

import os
import unittest


@unittest.skipUnless(
    os.environ.get("CAPYMOA_MOA_JAR"),
    "Set CAPYMOA_MOA_JAR to a MOA 2024.07 JAR with NA/GNG classes to run CapyMOA tests",
)
class TestCapyMOANA(unittest.TestCase):
    def test_prequential_smoke_sea_stream(self):
        try:
            from capymoa.evaluation import prequential_evaluation
            from capymoa.stream.generator import SEAStream
            from capymoa_na import NA
        except ImportError as exc:
            self.skipTest(f"CapyMOA not available: {exc}")

        stream = SEAStream(function=1, instance_random_seed=1)
        learner = NA(
            schema=stream.get_schema(),
            CLI="-l (trees.HoeffdingTree) -d (RDDM -o 2.0) -s Euclidean -L 0",
        )
        results = prequential_evaluation(
            stream=stream,
            learner=learner,
            max_instances=1000,
        )
        acc = results.cumulative.accuracy()
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 100.0)


if __name__ == "__main__":
    unittest.main()
