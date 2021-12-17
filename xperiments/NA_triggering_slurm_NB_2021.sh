#!/bin/bash -l
# see my jobs: squeue -u asuarezcet  / cancel a job: scancel <jobid>
# ls -l|awk '$8 ~ /..:../{gsub($8,"2020")}1'|awk '{A[$7" "$6" "$8]++}END{for (i in A){print i" "A[i]}}' # count of files by date in a dir
# find . -maxdepth 1 -type f -printf '%TY-%Tm-%Td Hour:%TH\n' | awk '{array[$0]+=1}END{ for(val in array) print val" "array[val]   }'  # including Hour
#
# Set only one task to perform an embarrassingly parallel workload
#SBATCH --ntasks=1
#SBATCH --array=0-15 # CPF
####SBATCH --array=0-895
###SBATCH --array=0-447
##SBATCH --array=0-1
###SBATCH --array=0-19
#### SBATCH --array=0-999 # for batches 0-3
####SBATCH --array=0-799 # for batch ==4
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
#SBATCH --output=/home/people/asuarezcet/job_outputs/slurm_output_NA_%J.txt
#
# Change working directory to current directory
cd $MOA_DEV/experiments/scripts

BATCH=0  # 0 for 0-999; 1 for 1000-1999; 2 for 2000-2999; 3 for 3000-3999; 4 for 4000=4799 (smaller array)

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
PROBLEMS="Synthetic_set_3"
# DATASET="synthetic_1607572321_train"
DATASET="synthetic_1607572321_devset"
# MAHABSET="1_equities_spy20200103T1407_indicators_best.arff;2_fixed_pref_pff20200106T1302_indicators_best.arff;3_real_state_vnq20200106T1215_indicators_best.arff;4_inter_bonds_bwx20200106T1531_indicators_best.arff"
MAHABSET="synthetic3_mahalanobis_state_1.arff;synthetic3_mahalanobis_state_2.arff;synthetic3_mahalanobis_state_3.arff;synthetic3_mahalanobis_state_4.arff"
# MAHABSET="1_equities_spy20200103T1407_indicators_best.arff;2_fixed_pref_pff20200106T1302_indicators_best.arff;3_real_state_vnq20200106T1215_indicators_best.arff;4_inter_bonds_bwx20200106T1531_indicators_best.arff"
# MAHABSET="synthetic_1607572321_before_dev"



#
# Params to multi-thread
ALGO=$1
THREAD=$(($SLURM_ARRAY_TASK_ID + 1000 * $BATCH))
# if [ $1 == "others" ]; then
#     ALGO=$2
#     if [ $2 == "base" ]; then
#         srun NA_loops_opt_NB_2021.sh ${ALGO} "" ${PROBLEMS} ${DATASET} "${MAHABSET}" 
#     fi
#     if [ $2 == "AUE" ]; then
#         srun NA_loops_opt_NB_2021.sh ${ALGO} "" ${PROBLEMS} ${DATASET} "${MAHABSET}" 
#     fi
#     if [ $2 == "ARF" ]; then
#         srun NA_loops_opt_NB_2021.sh ${ALGO} "" ${PROBLEMS} ${DATASET} "${MAHABSET}" 
#     fi
#     if [ $2 == "DWM" ]; then
#         srun NA_loops_opt_NB_2021.sh ${ALGO} "" ${PROBLEMS} ${DATASET} "${MAHABSET}" 
#     fi
# else
if [ $1 == "NA_1" ] || [ $1 == "NA_2" ] || [ $1 == "NA_1_2det" ] || [ $1 == "NA_2_2det" ]; then
    if [ $ALGO == "NA_1" ]  || [ $1 == "NA_1_2det" ]; then
        ARGS1=(1 1.25 1.5 1.75 2 2.25 2.5) # (0.5 0.75 1 1.25 1.5 1.75 2)
    fi
    if [ $ALGO == "NA_2" ] || [ $1 == "NA_2_2det" ]; then
        ARGS1=(1.5 1.75 2 2.25 2.5) # (0.5 0.75 1 1.25 1.5 1.75 2)
    fi
    ARGS2=(200) # (100 200 300 400 500)       
    ARGS3=("EDDM" "HDDM_A_Test" "ADDM" "RDDM")   #  "RDDM" "HDDM_A_Test" "ADWINChangeDetector" "HDDM_A_Test"  --array=0-104   0-34
    ARGS4=("-T")  # "'""'")  # TrainOnarningWindow
    ARGS5=("-B" "'""'") # Train after drift detection
    ARGS6=("N-n") # "'""'") # pretraining  (added N mayus to -n option so it does not dissapear.I deal with this in the next script)
    ARGS7=(5 10 15 20) #25)  ## 'w' warning window min threshold
    ARGS8=(200) # 'I'  100  300
    ARGS9=("-Q" "'""'") # insExonly

    # Fill array for concurrent tests
    declare -a my_array1
    declare -a my_array2
    declare -a my_array3
    declare -a my_array4
    declare -a my_array5
    declare -a my_array6
    declare -a my_array7
    declare -a my_array8

    for ARG1 in "${ARGS1[@]}"; do
    for ARG2 in "${ARGS2[@]}"; do
        for ARG3 in "${ARGS3[@]}"; do
        for ARG4 in "${ARGS4[@]}"; do
            for ARG5 in "${ARGS5[@]}"; do
            for ARG6 in "${ARGS6[@]}"; do
                for ARG7 in "${ARGS7[@]}"; do
                for ARG8 in "${ARGS8[@]}"; do
                for ARG9 in "${ARGS9[@]}"; do
                    my_array1+=($ARG1)
                    my_array2+=($ARG2)
                    my_array3+=($ARG3)
                    my_array4+=($ARG4)
                    my_array5+=($ARG5)
                    my_array6+=($ARG6)
                    my_array7+=($ARG7)
                    my_array8+=($ARG8)
                    my_array9+=($ARG9)
                done
                done
                done
            done
            done
        done
        done
    done
    done 
    
    ## ##########
    # -F always on and -Z always off (by now, till I discuss it with A&D)
    # -N?

    # Create the job (FOR INB I WASN'T PRINTING THE _INB STUFF. MIND THIS!!)
    echo "========= TASK ID: ${SLURM_ARRAY_TASK_ID} ========="
    echo "srun NA_loops_opt_NB_2021.sh ${ALGO} ${my_array1[$(($THREAD))]}  ${PROBLEMS} ${DATASET} "${MAHABSET}" ${my_array2[$(($THREAD))]}  ${my_array3[$(($THREAD))]} ${my_array4[$(($THREAD))]} ${my_array5[$(($THREAD))]} ${my_array6[$(($THREAD))]} ${my_array7[$(($THREAD))]} ${my_array8[$(($THREAD))]} ${my_array9[$(($THREAD))]}"
    # 240 * 5 parallel threads per feature subset.
    
    # echo "./run_parameter_optimization_NA.sh     srun run_parameter_optimization_NA.sh ${ALGO} ${ARGS1[$(($SLURM_ARRAY_TASK_ID%7))]} ${PROBLEMS} ${DATASET} "${MAHABSET}" ${ARGS2[$(($SLURM_ARRAY_TASK_ID%5))]}  ${ARGS3[$(($SLURM_ARRAY_TASK_ID%2))]}"
    echo "--------------------------"
    # srun run_parameter_optimization_NA.sh ${ALGO} ${ARGS1[$(($SLURM_ARRAY_TASK_ID%7))]} ${PROBLEMS} ${DATASET} ${MAHABSET} ${ARGS2[$(($SLURM_ARRAY_TASK_ID%5))]} ${ARGS3[$(($SLURM_ARRAY_TASK_ID%5))]}
    # TODO: add "" to wrap Mahalanobis sets?
    # srun run_parameter_optimization_NA.sh ${ALGO} ${ARGS1[$(($SLURM_ARRAY_TASK_ID%7))]} ${PROBLEMS} ${DATASET} "${MAHABSET}" ${ARGS2[$(($SLURM_ARRAY_TASK_ID%5))]} ${ARGS3[$(($SLURM_ARRAY_TASK_ID%2))]}
    srun NA_loops_opt_NB_2021.sh ${ALGO} ${my_array1[$(($THREAD))]}  ${PROBLEMS} ${DATASET} "${MAHABSET}" ${my_array2[$(($THREAD))]}  ${my_array3[$(($THREAD))]} "${my_array4[$(($THREAD))]}" "${my_array5[$(($THREAD))]}" "${my_array6[$(($THREAD))]}" ${my_array7[$(($THREAD))]} ${my_array8[$(($THREAD))]} ${my_array9[$(($THREAD))]}
elif [ $1 == "CPF" ]; then
    # Create the job
    CPF_ms=(0.925 0.95 0.975 0.99) # 0.9 0.85) 
    # CPF_min_buffer_sizes=(30 60 120)
    CPF_driftDets=("EDDM" "ADWINChangeDetector" "RDDM" "HDDM_A_Test")

    # Fill array for concurrent tests
    declare -a CPF_ms_array1
    # declare -a CPF_min_buffer_sizes_array2
    declare -a CPF_driftDets_array3
    for CPF_m in "${CPF_ms[@]}"; do
        # for CPF_min_buffer_size in "${CPF_min_buffer_sizes[@]}"; do
            for CPF_driftDet in "${CPF_driftDets[@]}"; do
                CPF_ms_array1+=($CPF_m)
                # CPF_min_buffer_sizes_array2+=($CPF_min_buffer_size)
                CPF_driftDets_array3+=($CPF_driftDet)
            done
        # done
    done 
    
    echo "========= TASK ID: ${SLURM_ARRAY_TASK_ID} ========="
    echo "./run_parameter_optimization_COMPETITORS_HT.sh ${ALGO} ${CPF_ms_array1[$(($SLURM_ARRAY_TASK_ID))]} ${PROBLEMS} ${DATASET} ${MAHABSET} ${CPF_driftDets_array3[$(($SLURM_ARRAY_TASK_ID))]}"
    echo "--------------------------"
    # srun run_parameter_optimization_COMPETITORS.sh ${ALGO} ${CPF_ms[$(($SLURM_ARRAY_TASK_ID%6))]} ${PROBLEMS} ${DATASET} ${MAHABSET} ${CPF_driftDets[$(($SLURM_ARRAY_TASK_ID%5))]}
    srun run_parameter_optimization_COMPETITORS_HT.sh ${ALGO} ${CPF_ms_array1[$(($SLURM_ARRAY_TASK_ID))]} ${PROBLEMS} ${DATASET} ${MAHABSET} ${CPF_driftDets_array3[$(($SLURM_ARRAY_TASK_ID))]}

    # ARGS=(0.5 0.75 1 1.25 1.5 1.75 2)
    # echo "========= TASK ID: ${SLURM_ARRAY_TASK_ID} ========="
    # echo "./run_parameter_optimization_COMPETITORS.sh ${ALGO} ${ARGS[${SLURM_ARRAY_TASK_ID}]} ${PROBLEMS} ${DATASET} ${MAHABSET}"
    # echo "--------------------------"
    # # TODO: modify to NA_loops_opt.sh?
    # srun run_parameter_optimization_COMPETITORS.sh ${ALGO} ${ARGS[$SLURM_ARRAY_TASK_ID]} ${PROBLEMS} ${DATASET} ${MAHABSET}
elif [ $1 == "RCD" ]; then
    # Create the job
    RCD_bs=(100 200 300 400 500) 
    # RCD_driftDets=("DDM" "EDDM" "ADWINChangeDetector" "RDDM" "HDDM_A_Test")
    RCD_driftDets=("EDDM" "ADDM" "RDDM" "HDDM_A_Test")
    echo "========= TASK ID: ${SLURM_ARRAY_TASK_ID} ========="
    echo "./run_parameter_optimization_COMPETITORS.sh ${ALGO} ${RCD_bs[$(($SLURM_ARRAY_TASK_ID%5))]} ${PROBLEMS} ${DATASET} ${MAHABSET} ${RCD_driftDets[$(($SLURM_ARRAY_TASK_ID%5))]}"
    echo "--------------------------"
    srun run_parameter_optimization_COMPETITORS.sh ${ALGO} ${RCD_bs[$(($SLURM_ARRAY_TASK_ID%5))]} ${PROBLEMS} ${DATASET} ${MAHABSET} ${RCD_driftDets[$(($SLURM_ARRAY_TASK_ID%5))]}
    # srun run_parameter_optimization_COMPETITORS_HT.sh ${ALGO} ${RCD_bs[$(($SLURM_ARRAY_TASK_ID%5))]} ${PROBLEMS} ${DATASET} ${MAHABSET} ${RCD_driftDets[$(($SLURM_ARRAY_TASK_ID%5))]}
elif [ $1 == "DWM" ]; then
    # Create the job
    DWM_periods=(1 25 50 75 100) 
    DWM_thetas=(0.005 0.01 0.025 0.05)
    echo "========= TASK ID: ${SLURM_ARRAY_TASK_ID} ========="
    echo "./run_parameter_optimization_COMPETITORS.sh ${ALGO} ${DWM_periods[$(($SLURM_ARRAY_TASK_ID%5))]} ${PROBLEMS} ${DATASET} ${MAHABSET} ${DWM_thetas[$(($SLURM_ARRAY_TASK_ID%4))]}"
    echo "--------------------------"
    srun run_parameter_optimization_COMPETITORS.sh ${ALGO} ${DWM_periods[$(($SLURM_ARRAY_TASK_ID%5))]} ${PROBLEMS} ${DATASET} ${MAHABSET} ${DWM_thetas[$(($SLURM_ARRAY_TASK_ID%4))]}
    # srun run_parameter_optimization_COMPETITORS_HT.sh ${ALGO} ${DWM_periods[$(($SLURM_ARRAY_TASK_ID%5))]} ${PROBLEMS} ${DATASET} ${MAHABSET} ${DWM_thetas[$(($SLURM_ARRAY_TASK_ID%4))]}

elif [ $1 == "LNSE" ]; then
    # Create the job
    LNSE_periods=(10 20 40 60 80 100) 
    LNSE_sigSlopes=(5 10 15 20)
    echo "========= TASK ID: ${SLURM_ARRAY_TASK_ID} ========="
    echo "./run_parameter_optimization_COMPETITORS.sh ${ALGO} ${LNSE_periods[$(($SLURM_ARRAY_TASK_ID%6))]} ${PROBLEMS} ${DATASET} ${MAHABSET} ${LNSE_sigSlopes[$(($SLURM_ARRAY_TASK_ID%4))]}"
    echo "--------------------------"
    srun run_parameter_optimization_COMPETITORS.sh ${ALGO} ${LNSE_periods[$(($SLURM_ARRAY_TASK_ID%6))]} ${PROBLEMS} ${DATASET} ${MAHABSET} ${LNSE_sigSlopes[$(($SLURM_ARRAY_TASK_ID%4))]}
else
    # Create the job
    ARGS=("ARF" "AUE" "base")
    echo "========= TASK ID: ${SLURM_ARRAY_TASK_ID} ========="
    echo "./run_parameter_optimization_COMPETITORS.sh ${ARGS[${SLURM_ARRAY_TASK_ID}]} null ${PROBLEMS} ${DATASET} ${MAHABSET}"
    echo "--------------------------"
    srun run_parameter_optimization_COMPETITORS.sh ${ARGS[${SLURM_ARRAY_TASK_ID}]} "null" ${PROBLEMS} ${DATASET} ${MAHABSET}
    # srun run_parameter_optimization_COMPETITORS_HT.sh ${ARGS[${SLURM_ARRAY_TASK_ID}]} "null" ${PROBLEMS} ${DATASET} ${MAHABSET}
fi
# fi
# TEST
# $MOA_DEV/experiments/scripts/run_parameter_optimization_NA.sh NA_1 1 Synthethic_TS_new 3_timeseries_created_1576000391_indicators APPLE_[2018-08-01_to_2018-09-11]_5min_indicators_best.arff;BITCOIN_[2019-07-01_to_2019-07-15]_5min_indicators_best.arff;RIPPLE_[2019-07-01_to_2019-08-01]_5min_indicators_best.arff;DOWJONES_[2015-08-01_to_2015-08-31]_market_hours_indicators_best.arff 200 ADWINChangeDetector

