#!/usr/bin/env bash
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
#
# Build a custom CapyMOA fat JAR containing the GroCH + GNG MOA classes.
#
# Usage:
#   ./scripts/build_moa_jar.sh [output_jar_path] [python_bin]
#
# Example:
#   ./scripts/build_moa_jar.sh ./moa_with_groch.jar python3
#   export CAPYMOA_MOA_JAR="$(pwd)/moa_with_groch.jar"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_JAR="${1:-${ROOT_DIR}/moa_with_groch.jar}"
PYTHON_BIN="${2:-python3}"

BASE_MOA_JAR="$("${PYTHON_BIN}" -c 'import capymoa, pathlib; print(pathlib.Path(capymoa.__file__).parent / "jar" / "moa.jar")')"
if [[ ! -f "${BASE_MOA_JAR}" ]]; then
  echo "ERROR: Could not locate CapyMOA's bundled moa.jar at ${BASE_MOA_JAR}" >&2
  exit 1
fi

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "${BUILD_DIR}"' EXIT

JAVA_SRC_ROOT="${ROOT_DIR}/src"
JAVA_FILES=(
  "${JAVA_SRC_ROOT}/moa/classifiers/core/driftdetection/ADDM.java"
  "${JAVA_SRC_ROOT}/moa/classifiers/core/driftdetection/GroCHChangeDetector.java"
  "${JAVA_SRC_ROOT}/moa/classifiers/lazy/neighboursearch/EuclideanDistanceModified.java"
  "${JAVA_SRC_ROOT}/moa/classifiers/lazy/neighboursearch/LinearNNSearchModified.java"
  "${JAVA_SRC_ROOT}/moa/classifiers/meta/ConceptDistance.java"
  "${JAVA_SRC_ROOT}/moa/classifiers/meta/DriftEventLogger.java"
  "${JAVA_SRC_ROOT}/moa/classifiers/meta/EuclideanConceptDistance.java"
  "${JAVA_SRC_ROOT}/moa/classifiers/meta/MahalanobisDistanceSingleMatrix.java"
  "${JAVA_SRC_ROOT}/moa/classifiers/meta/PerGroupMahalanobisDistance.java"
  "${JAVA_SRC_ROOT}/moa/classifiers/meta/GroCH.java"
  "${JAVA_SRC_ROOT}/moa/clusterers/gng/GEdge.java"
  "${JAVA_SRC_ROOT}/moa/clusterers/gng/GUnit.java"
  "${JAVA_SRC_ROOT}/moa/clusterers/gng/GNG.java"
  "${JAVA_SRC_ROOT}/moa/evaluation/DynamicWindowClassificationPerformanceEvaluator.java"
)

echo "[build_moa_jar] Compiling ${#JAVA_FILES[@]} Java files against ${BASE_MOA_JAR}..."
javac -encoding ISO-8859-1 \
  -cp "${BASE_MOA_JAR}" \
  -d "${BUILD_DIR}" \
  "${JAVA_FILES[@]}"

echo "[build_moa_jar] Packaging ${OUT_JAR}..."
cp "${BASE_MOA_JAR}" "${OUT_JAR}"
jar uf "${OUT_JAR}" -C "${BUILD_DIR}" .

# Remove stale JAR index so the JVM classloader discovers newly added packages.
if command -v zip >/dev/null 2>&1; then
  zip -d "${OUT_JAR}" META-INF/INDEX.LIST >/dev/null 2>&1 || true
fi

echo "[build_moa_jar] Done!"
echo "  export CAPYMOA_MOA_JAR=\"${OUT_JAR}\""
