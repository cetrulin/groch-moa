#!/bin/bash -l
# see my jobs: squeue -u asuarezcet  / cancel a job: scancel <jobid>
# ls -l|awk '$8 ~ /..:../{gsub($8,"2020")}1'|awk '{A[$7" "$6" "$8]++}END{for (i in A){print i" "A[i]}}' # count of files by date in a dir
# find . -maxdepth 1 -type f -printf '%TY-%Tm-%Td Hour:%TH\n' | awk '{array[$0]+=1}END{ for(val in array) print val" "array[val]   }'  # including Hour
#
# Set only one task to perform an embarrassingly parallel workload
#SBATCH --ntasks=1
#SBATCH --array=1-20 
#####SBATCH --array=7-7 

####SBATCH --array=0-1 FOR BASE JUST LEAVE ALL ARRAYS COMMENTED OUT SO THERE IS NONE
##SBATCH --array=0-19   # new NA per SEED
###### SBATCH --array=0-69 
###  174  (for the 5 detectors)
#
# E-mail on begin (b), abort (a) and end (e) of job
#SBATCH --mail-type=ALL
#
# Set the walltime of the job to 48 hours (format is hh:mm:ss)
############SBATCH -t 48:00:00
#
# E-mail address of recipient
#SBATCH --mail-user=100080038@alumnos.uc3m.es
#
# Specifies the jobname
#SBATCH --output=/home/people/asuarezcet/job_outputs/final_SPY_%J.txt
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

PROBLEMS=${5}  # "spy_seeds"
DATASET="spy_train"
# DATASET="spydevset"
# MAHABSET="spy_mahalanobis_state_1.arff;spy_mahalanobis_state_2.arff;spy_mahalanobis_state_3.arff"

# Change working directory to scripts directory
cd $MOA_DEV/experiments/drifts/analyse_results

echo "trigger Python script"
# Fill array of best params
# TODO? 

echo "If src files for next scripts are missing, run: python -m src.analysis_PO_seeds -D ${2}"
# python -m src.analysis_PO_seeds -D ${2}  # this can be slow

echo "python -m src.analysis_selection_of_best_results -D ${2}"
python -m src.analysis_selection_of_best_results -D ${2}

lines=($(python -m src.trigger_best_results -D $2 -S ${SLURM_ARRAY_TASK_ID} -B $3 -A $1)) #> $lines
declare -a my_array1
echo "python -m src.trigger_best_results -D ${2} -S ${SLURM_ARRAY_TASK_ID} -B ${3} -A ${1}"
# echo $lines
echo "########"
echo "best params:"
# #######
# echo ${lines[0]}
# echo ${lines[1]}
# #########
# while read line ; do
for line in "${lines[@]}"; do
#     my_array1+=($line)
    echo $line
#     echo ${my_array1[0]}
done 


# Change working directory to current directory
# cd $MOA_DEV/experiments/scripts

# Create the job
echo "========= TASK ID: ${SLURM_ARRAY_TASK_ID} ========="
# echo "srun NA_loops_opt_NB_2021.sh ${ALGO} ${my_array1[$(($THREAD))]}  ${PROBLEMS} ${DATASET} "${MAHABSET}" ${my_array2[$(($THREAD))]}  ${my_array3[$(($THREAD))]} ${my_array4[$(($THREAD))]} ${my_array5[$(($THREAD))]} ${my_array6[$(($THREAD))]} ${my_array7[$(($THREAD))]} ${my_array8[$(($THREAD))]} ${my_array9[$(($THREAD))]}"
# 240 * 5 parallel threads per feature subset.
echo "srun ${MOA_DEV}/experiments/scripts/run_spy_tests_May2021.sh ${1} ${2} ${SLURM_ARRAY_TASK_ID} ${3} ${4} ${lines[0]} ${lines[1]} ${lines[2]} ${lines[3]} ${lines[4]}"
echo "--------------------------"
srun $MOA_DEV/experiments/scripts/run_spy_tests_May2021.sh ${1} ${2} ${SLURM_ARRAY_TASK_ID} ${3} ${4} ${lines[0]} ${lines[1]} ${lines[2]} ${lines[3]} ${lines[4]}

# Example to call this
# sbatch trigger_tests_SPY_seeds_May2021.sh NA spy-seeds-5min 1 HT spy_seeds_5m

# TEST (to be updated)
# $MOA_DEV/experiments/scripts/run_parameter_optimization_NA.sh NA 1 Synthethic_TS_new 3_timeseries_created_1576000391_indicators APPLE_[2018-08-01_to_2018-09-11]_5min_indicators_best.arff;BITCOIN_[2019-07-01_to_2019-07-15]_5min_indicators_best.arff;RIPPLE_[2019-07-01_to_2019-08-01]_5min_indicators_best.arff;DOWJONES_[2015-08-01_to_2015-08-31]_market_hours_indicators_best.arff 200 ADWINChangeDetector

#!/bin/bash
# script for tesing

