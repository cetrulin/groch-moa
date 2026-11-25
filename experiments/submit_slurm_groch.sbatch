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
# SLURM job-array template for running GroCH sweeps in parallel.
#
# Usage:
#   MOA_JAR=/path/to/moa_with_groch.jar \
#   ARFF=/path/to/data.arff \
#   OUT_DIR=/path/to/results \
#   sbatch experiments/submit_slurm_groch.sbatch

#SBATCH --job-name=groch_moa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=1-10
#SBATCH --output=logs/groch_%A_%a.out
#SBATCH --error=logs/groch_%A_%a.err
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your.email@example.com

set -euo pipefail

: "${MOA_JAR:?Set MOA_JAR before submitting}"
: "${ARFF:?Set ARFF before submitting}"
OUT_DIR="${OUT_DIR:-./results}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p logs "${OUT_DIR}"

"${SCRIPT_DIR}/run_groch.sh" \
  --arff "${ARFF}" \
  --out "${OUT_DIR}/groch_task_${TASK_ID}.csv" \
  "$@"
