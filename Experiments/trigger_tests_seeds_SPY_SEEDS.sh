#!/bin/bash -l
# see my jobs: squeue -u asuarezcet  / cancel a job: scancel <jobid>
# ls -l|awk '$8 ~ /..:../{gsub($8,"2020")}1'|awk '{A[$7" "$6" "$8]++}END{for (i in A){print i" "A[i]}}' # count of files by date in a dir
# find . -maxdepth 1 -type f -printf '%TY-%Tm-%Td Hour:%TH\n' | awk '{array[$0]+=1}END{ for(val in array) print val" "array[val]   }'  # including Hour
#
# Set only one task to perform an embarrassingly parallel workload
######SBATCH --ntasks=1
#####SBATCH --array=0-1
#
# E-mail on begin (b), abort (a) and end (e) of job
#SBATCH --mail-type=ALL
#
#SBATCH --job-name=seeds_$1
#
# Set the walltime of the job to 48 hours (format is hh:mm:ss)
############SBATCH -t 48:00:00
#
# E-mail address of recipient
#SBATCH --mail-user=100080038@alumnos.uc3m.es
#
# Specifies the jobname
#SBATCH --output=/home/people/asuarezcet/job_outputs/slurm_output_NA_final_%J.txt
#
# Change working directory to current directory
cd $MOA_DEV/experiments/scripts

#
# Dataset
# PROBLEMS="real_crypto"
# DATASET="BTC_dev_set_1min_best"
# MAHABSET="BITCOIN_Selection_Mahalanobis_set_[2018-07-01_to_2018-10-01]_1min_indicators_best.arff"
# PROBLEMS="real_spy_14_years"
# DATASET="spy_historical_1min_dev_indicators_best"
# MAHABSET="spy_historical_1min_mahalanobis_indicators_best.arff"
# PROBLEMS="Syntethic_TS_new"
# DATASET="3_timeseries_created_1576000391_indicators"
# MAHABSET="APPLE_[2018-08-01_to_2018-09-11]_5min_indicators_best.arff;BITCOIN_[2019-07-01_to_2019-07-15]_5min_indicators_best.arff;RIPPLE_[2019-07-01_to_2019-08-01]_5min_indicators_best.arff;DOWJONES_[2015-08-01_to_2015-08-31]_market_hours_indicators_best.arff"
# PROBLEMS="Synthetic_set_2"
# # DATASET="synthetic_set_2"
# DATASET="synthetic_set_2_devset"
# MAHABSET="biv20170210T1354.arff;spxl20180524T1550.arff;iwo20170607T1613.arff;vv20171101T1523.arff"
###
PROBLEMS="spy_seeds_"$5
DATASET="spy_train"
# DATASET="spy_devset"
MAHABSET="spy_mahalanobis_state_1.arff;spy_mahalanobis_state_2.arff;spy_mahalanobis_state_3.arff"


# Create the job (FOR INB I WASN'T PRINTING THE _INB STUFF. MIND THIS!!)
echo "========= TASK ID: ${SLURM_ARRAY_TASK_ID} ========="
# echo "srun NA_loops_opt_NB_2021.sh ${ALGO} ${my_array1[$(($THREAD))]}  ${PROBLEMS} ${DATASET} "${MAHABSET}" ${my_array2[$(($THREAD))]}  ${my_array3[$(($THREAD))]} ${my_array4[$(($THREAD))]} ${my_array5[$(($THREAD))]} ${my_array6[$(($THREAD))]} ${my_array7[$(($THREAD))]} ${my_array8[$(($THREAD))]} ${my_array9[$(($THREAD))]}"
# 240 * 5 parallel threads per feature subset.
echo "srun run_tests_SPY_SEEDS.sh ${1} ${2} ${3} ${4} ${PROBLEMS} ${DATASET}"
# echo "./run_parameter_optimization_NA.sh     srun run_parameter_optimization_NA.sh ${ALGO} ${ARGS1[$(($SLURM_ARRAY_TASK_ID%7))]} ${PROBLEMS} ${DATASET} "${MAHABSET}" ${ARGS2[$(($SLURM_ARRAY_TASK_ID%5))]}  ${ARGS3[$(($SLURM_ARRAY_TASK_ID%2))]}"
echo "--------------------------"
# srun run_parameter_optimization_NA.sh ${ALGO} ${ARGS1[$(($SLURM_ARRAY_TASK_ID%7))]} ${PROBLEMS} ${DATASET} ${MAHABSET} ${ARGS2[$(($SLURM_ARRAY_TASK_ID%5))]} ${ARGS3[$(($SLURM_ARRAY_TASK_ID%5))]}
# TODO: add "" to wrap Mahalanobis sets?
# srun run_parameter_optimization_NA.sh ${ALGO} ${ARGS1[$(($SLURM_ARRAY_TASK_ID%7))]} ${PROBLEMS} ${DATASET} "${MAHABSET}" ${ARGS2[$(($SLURM_ARRAY_TASK_ID%5))]} ${ARGS3[$(($SLURM_ARRAY_TASK_ID%2))]}
# srun run_tests_2021-02-08.sh ${ALGO} ${my_array1[$(($THREAD))]}  ${PROBLEMS} ${DATASET} "${MAHABSET}" ${my_array2[$(($THREAD))]}  ${my_array3[$(($THREAD))]} "${my_array4[$(($THREAD))]}" "${my_array5[$(($THREAD))]}" "${my_array6[$(($THREAD))]}" ${my_array7[$(($THREAD))]} ${my_array8[$(($THREAD))]} ${my_array9[$(($THREAD))]}
srun ./run_tests_SPY_SEEDS.sh $1 $2 $3 $4 ${PROBLEMS} ${DATASET}
# NA_tests_HT_best_config_EDDM_10Feb test12
# srun run_tests_2021-02-08.sh NA_tests_HT_best_config_10Feb_test12_ww20 test12
# srun run_tests_2021-02-08.sh tests_dist5_ww20_insertExamplesOnly_10Feb_test12_ww20 test12
# srun run_tests_2021-02-08.sh tests_dist5_ww20_insertExamplesOnly_10Feb_EDDM_ww5 test12
# srun run_tests_2021-02-08.sh NA_tests_NB_best_config_10022021 test12
# srun run_tests_2021-02-08.sh tests_insertExamplesOnly_notInsED_NB_10022021 test12

# TEST
# $MOA_DEV/experiments/scripts/run_parameter_optimization_NA.sh NA_1 1 Synthethic_TS_new 3_timeseries_created_1576000391_indicators APPLE_[2018-08-01_to_2018-09-11]_5min_indicators_best.arff;BITCOIN_[2019-07-01_to_2019-07-15]_5min_indicators_best.arff;RIPPLE_[2019-07-01_to_2019-08-01]_5min_indicators_best.arff;DOWJONES_[2015-08-01_to_2015-08-31]_market_hours_indicators_best.arff 200 ADWINChangeDetector

