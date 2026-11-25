# GroCH Experiment Runners (Bash & SLURM)

Parameterized runners for `meta.GroCH` on MOA 2024.07:

- `run_groch.sh`: runs `EvaluateInterleavedTestThenTrain` on an ARFF stream (supports `--dry-run`).
- `submit_slurm_groch.sbatch`: SLURM job-array wrapper calling `run_groch.sh`.

### Local

```bash
./scripts/build_moa_jar.sh ./moa_with_groch.jar python3

MOA_JAR=./moa_with_groch.jar ./experiments/run_groch.sh \
  --arff path/to/data.arff \
  --base-learner "trees.HoeffdingTree" \
  --drift "RDDM -o 2.0" \
  --distance "Euclidean" \
  --out ./results/groch_run.csv
```

### SLURM

```bash
MOA_JAR="$(pwd)/moa_with_groch.jar" ARFF="/path/to/data.arff" OUT_DIR="$(pwd)/results" \
  sbatch experiments/submit_slurm_groch.sbatch --distance Mahalanobis
```

The original 2021 MOA 2017.06 `meta.NA` thesis sweep scripts are preserved verbatim on tags `v0.1-moa2017.06` and `v1.0.0-moa2017.06`.
