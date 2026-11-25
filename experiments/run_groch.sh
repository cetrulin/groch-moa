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
# Parameterized runner for GroCH (meta.GroCH) on MOA 2024.07.
#
# Usage:
#   MOA_JAR=./moa_with_groch.jar ./experiments/run_groch.sh \
#     --arff path/to/data.arff [--out results.csv] [--dry-run]
#
# Environment / Option overrides:
#   MOA_JAR       Path to MOA fat JAR containing compiled GroCH & GNG classes (required)
#   JAVA_BIN      Java executable (default: java)
#   JAVA_OPTS     Extra JVM options (default: -Xmx4g)
#   SIZEOF_JAR    Optional path to sizeofag JAR for memory profiling

set -euo pipefail

ARFF=""
OUT=""
BASE_LEARNER="trees.HoeffdingTree"
DRIFT_DETECTOR="RDDM -o 2.0"
DISTANCE="Euclidean"
GNG_OPTS="GNG -l 5 -m 200 -c 2147483647 -b"
GROCH_FLAGS="-F -T -B -w 20 -a 1 -D 1 -z 1 -I 100 -M 5 -t 1.0 -P FIFO -V 0"
MAX_INSTANCES="-1"
SAMPLE_FREQ="1000"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arff)          ARFF="$2"; shift 2 ;;
    --out)           OUT="$2"; shift 2 ;;
    --base-learner)  BASE_LEARNER="$2"; shift 2 ;;
    --drift)         DRIFT_DETECTOR="$2"; shift 2 ;;
    --distance)      DISTANCE="$2"; shift 2 ;;
    --gng)           GNG_OPTS="$2"; shift 2 ;;
    --groch-flags)   GROCH_FLAGS="$2"; shift 2 ;;
    --max-instances) MAX_INSTANCES="$2"; shift 2 ;;
    --sample-freq)   SAMPLE_FREQ="$2"; shift 2 ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '14,24p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

: "${MOA_JAR:?Set MOA_JAR to a MOA 2024.07 JAR containing compiled GroCH classes}"
: "${ARFF:?Provide --arff path/to/dataset.arff}"

JAVA_BIN="${JAVA_BIN:-java}"
JAVA_OPTS="${JAVA_OPTS:--Xmx4g}"

JVM_ARGS=()
read -r -a _opts <<< "${JAVA_OPTS}"
JVM_ARGS+=("${_opts[@]}")
if [[ -n "${SIZEOF_JAR:-}" ]]; then
  JVM_ARGS+=("-javaagent:${SIZEOF_JAR}")
fi
JVM_ARGS+=("-cp" "${MOA_JAR}" "moa.DoTask")

TASK="EvaluateInterleavedTestThenTrain \
-l (meta.GroCH -l (${BASE_LEARNER}) -d (${DRIFT_DETECTOR}) -c (${GNG_OPTS}) -s ${DISTANCE} ${GROCH_FLAGS}) \
-s (ArffFileStream -f ${ARFF}) \
-e (BasicClassificationPerformanceEvaluator) \
-i ${MAX_INSTANCES} -f ${SAMPLE_FREQ}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%q ' "${JAVA_BIN}" "${JVM_ARGS[@]}" "${TASK}"
  echo
  exit 0
fi

if [[ -n "${OUT}" ]]; then
  mkdir -p "$(dirname "${OUT}")"
  "${JAVA_BIN}" "${JVM_ARGS[@]}" "${TASK}" > "${OUT}"
else
  "${JAVA_BIN}" "${JVM_ARGS[@]}" "${TASK}"
fi
