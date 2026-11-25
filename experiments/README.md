# Running GroCH Experiments (Bash & SLURM)

This directory contains clean, parameterized runners for **`meta.GroCH`** on **MOA 2024.07**:

* **`run_groch.sh`**: runs `EvaluateInterleavedTestThenTrain` with `meta.GroCH` and configurable base learner, drift detector, distance metric, GNG parameters, and GroCH flags.
* **`submit_slurm_groch.sbatch`**: portable SLURM job-array template delegating to `run_groch.sh`.

### Local Example

```bash
# Build a MOA 2024.07 fat JAR with GroCH & GNG first:
./scripts/build_moa_jar.sh ./moa_with_groch.jar python3

# Run GroCH on an ARFF stream:
MOA_JAR=./moa_with_groch.jar ./experiments/run_groch.sh \
  --arff path/to/data.arff \
  --base-learner "trees.HoeffdingTree" \
  --drift "RDDM -o 2.0" \
  --distance "Euclidean" \
  --out ./results/groch_run.csv
```

### SLURM Example

```bash
MOA_JAR="$(pwd)/moa_with_groch.jar" \
ARFF="/path/to/data.arff" \
OUT_DIR="$(pwd)/results" \
sbatch experiments/submit_slurm_groch.sbatch --distance Mahalanobis
```

> **Historical 2021 Thesis Scripts:** The original 2021 MOA 2017.06 `meta.NA` parameter-sweep scripts used in the PhD thesis and *Neurocomputing* (2026) paper are preserved verbatim under tags **`v0.1-moa2017.06`** and **`v1.0.0-moa2017.06`**.
