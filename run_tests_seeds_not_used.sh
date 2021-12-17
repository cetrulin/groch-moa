#!/bin/bash
# MOA_DEV=/home/bin/moa/src/moa-2017.06-sources
WEKALOC=${MOA_DEV}/weka
MOALOC=${MOA_DEV}/moa
JAVAAGENT=${MOA_DEV}/sizeofag-1.0.0.jar
#EXTRAPATH="./lib/colt.jar:../moaAlgorithms/classes:../wekaAlgorithms/classes:../weka-3-7-4/lib/libsvm.jar:/home/alex/src/gpsc/bin"
#EXTRAPATH="./lib/colt.jar:../moaAlgorithms/classes:../wekaAlgorithms/classes:../weka-3-7-4/lib/libsvm.jar:../gpsc/bin:../RCARF-master/bin"
EXTRAPATH="${MOA_DEV}/lib/*":"${MOA_DEV}/moa_classes/classifiers"
RESULTPATH=${RESULTS_PATH}
DATAPATH=${DATA_PATH}"/synthetic"
TIMESTAMP="$(date +"%T")" 

# #########################
# Define experiment
# #########################

# Datasets

# Synthetic set 1
# PROBLEM="Synthethic_TS_new"
# DATASET="3_timeseries_created_1576000391_indicators"
# DATASETPATH=${DATAPATH}/${PROBLEM}
# MAHABSET="${DATASETPATH}/APPLE_[2018-08-01_to_2018-09-11]_5min_indicators_best.arff;${DATASETPATH}/BITCOIN_[2019-07-01_to_2019-07-15]_5min_indicators_best.arff;${DATASETPATH}/RIPPLE_[2019-07-01_to_2019-08-01]_5min_indicators_best.arff;${DATASETPATH}/DOWJONES_[2015-08-01_to_2015-08-31]_market_hours_indicators_best.arff"

# Synthetic set 2
# PROBLEM="Synthetic_set_2"
# DATASET="synthetic_set_2"
# DATASETPATH=${DATAPATH}/${PROBLEM}
# MAHABSET="${DATASETPATH}/biv20170210T1354.arff;${DATASETPATH}/spxl20180524T1550.arff;${DATASETPATH}/iwo20170607T1613.arff;${DATASETPATH}/vv20171101T1523.arff"

# Synthetic set 3 SEEDS
PROBLEM="synthetic3_seeds"
DATASET="synthetic3_train"
#DATASET="synthetic3_devset"
SEED=$3
DATASETPATH=${DATAPATH}/${PROBLEM}/${SEED}
# raw sets for mahalanobis
# maraw="maraw_" # ""
# generated states for mahalanobis
maraw=""
MAHABSET="${DATASETPATH}/synthetic3_mahalanobis_state_1.arff;${DATASETPATH}/synthetic3_mahalanobis_state_2.arff;${DATASETPATH}/synthetic3_mahalanobis_state_3.arff;${DATASETPATH}/synthetic3_mahalanobis_state_4.arff"

# Instances and evaluation
WSIZE=1000
BATCHSIZE=10
# MAX_INSTANCES=1000000 # 2000000 1000000 350000
# distThresh=5  # 1 for subtop1 by default
# lambda=5
MAX_INSTANCES=500000 # 500000 1000000 

INB="bayes.NaiveBayes"
HT="trees.HoeffdingTree"
NA_baseClassifier=$HT  # $INB
NA_baseClassifier_name="HT"   # NB

distThresh=5  # 1 for subtop1 by default
lambda=5  # 200


# Params NA
subtop1="1,3,12,15,16,17"
subtop2="0,1,3,5,6,12,15,16,17"

testTTWindow() {
  ## Paths
  TRAINDATA=${DATASETPATH}/${DATASET}.arff
  RESULTFOLDER=${RESULTPATH}"/"${PROBLEM}"/"${SEED}
  mkdir ${RESULTFOLDER}
  EVENTS_FILE=$RESULTFOLDER'/'$PROBLEM'_'$ALGOTOK'_EVENTS_w'${WSIZE}
  rm ${RESULTFOLDER}/${DATASET}_${ALGOTOK}_w${WSIZE}_dump.txt

  # #########################
  # Run experiment
  # #########################
  java -cp $EXTRAPATH:$MOA_DEV -javaagent:$JAVAAGENT moa.DoTask \
  "EvaluateInterleavedTestThenTrain \
    -l (${ALGORITHM} -e ${EVENTS_FILE}.txt) \
    -s (ArffFileStream -f ${TRAINDATA}) \
    -i ${MAX_INSTANCES} \
    -r ${SEED} \
    -e (WindowClassificationPerformanceEvaluator -w ${WSIZE}) \
    -d ${RESULTFOLDER}/${DATASET}_${ALGOTOK}_w${WSIZE}_dump.txt \
    -f ${BATCHSIZE} " \
  > ${RESULTFOLDER}/${DATASET}_${ALGOTOK}_w${WSIZE}.txt
  #   
  cd $RESULTFOLDER
  zip $PROBLEM'_'$ALGOTOK'_EVENTS_w'${WSIZE}'.zip' $PROBLEM'_'$ALGOTOK'_EVENTS_w'${WSIZE}'.txt'
  rm $EVENTS_FILE'.txt'
  cd $MOA_DEV
  echo 'RUN:    tail -n 50 ${RESULTFOLDER}/${DATASET}_${ALGOTOK}_w${WSIZE}.txt'
}

testTTWindowTerm() {
  ## Paths
  TRAINDATA=${DATASETPATH}/${DATASET}.arff
  RESULTFOLDER=${RESULTPATH}"/"${PROBLEM}"/"${SEED}
  mkdir ${RESULTFOLDER}
  EVENTS_FILE=$RESULTFOLDER'/'$PROBLEM'_'$ALGOTOK'_EVENTS_w'${WSIZE}
  rm ${RESULTFOLDER}/${DATASET}_${ALGOTOK}_w${WSIZE}_dump.txt
  rm ${RESULTFOLDER}/$EVENTS_FILE'.txt'

  # #########################
  # Run experiment
  # #########################
  java -cp $EXTRAPATH:$MOA_DEV -javaagent:$JAVAAGENT moa.DoTask \
  "EvaluateInterleavedTestThenTrain \
    -l (${ALGORITHM} -e ${EVENTS_FILE}.txt) \
    -s (ArffFileStream -f ${TRAINDATA}) \
    -i ${MAX_INSTANCES} \
    -r ${SEED} \
    -e (WindowClassificationPerformanceEvaluator -w ${WSIZE}) \
    -f ${BATCHSIZE} " #\
#   > ${RESULTFOLDER}/${DATASET}_${ALGOTOK}_w${WSIZE}.txt
  #   -d ${RESULTFOLDER}/${DATASET}_${ALGOTOK}_w${WSIZE}_dump.txt \
#   cd $RESULTFOLDER
#   zip $PROBLEM'_'$ALGOTOK'_EVENTS_w'${WSIZE}'.zip' $PROBLEM'_'$ALGOTOK'_EVENTS_w'${WSIZE}'.txt'
#   rm $EVENTS_FILE'.txt'
#   cd $MOA_DEV
#   echo 'RUN:    tail -n 50 ${RESULTFOLDER}/${DATASET}_${ALGOTOK}_w${WSIZE}.txt'
}

testTTWindowOthers() {
  ## Paths
  TRAINDATA=${DATASETPATH}/${DATASET}.arff
  RESULTFOLDER=${RESULTPATH}"/"${PROBLEM}"/"${SEED}
  mkdir ${RESULTFOLDER}
  rm ${RESULTFOLDER}/${DATASET}_${ALGOTOK}_w${WSIZE}_dump.txt

  # #########################
  # Run experiment
  # #########################
  java -cp $EXTRAPATH:$MOA_DEV -javaagent:$JAVAAGENT moa.DoTask \
  "EvaluateInterleavedTestThenTrain \
    -l (${ALGORITHM}) \
    -s (ArffFileStream -f ${TRAINDATA}) \
    -i ${MAX_INSTANCES} \
    -r ${SEED} \
    -e (WindowClassificationPerformanceEvaluator -w ${WSIZE}) \
    -f ${BATCHSIZE} " \
  > ${RESULTFOLDER}/${DATASET}_${ALGOTOK}_w${WSIZE}.txt
  #   -d ${RESULTFOLDER}/${DATASET}_${ALGOTOK}_w${WSIZE}_dump.txt \
  cd $MOA_DEV
  echo 'RUN:    tail -n 50 ${RESULTFOLDER}/${DATASET}_${ALGOTOK}_w${WSIZE}.txt'
}

# ##############################
# Variants of NA:
# ======================
# trainEND         -> 'B'
# TestBKG          -> 'K'
# TrainActOnWarn   -> 'T'
# forceEarlyDrifts -> 'F'


updateTops="-j" # " " "-j"  # blank to not update topologies
test0="RDDM -w 1.773 -o 2.258" # default
test1="RDDM -w 2.258 -o 2.743"
test2="RDDM -w 2.258 -o 5"
test3="RDDM -w 2.258 -o 3.5"
# I was doing the opposite way around. 
test4="RDDM -w 1.288 -o 1.773"
# good now...
test5="RDDM -w 1.288 -o 5"
test6="RDDM -w 1.288 -o 3.5"
test7="RDDM -w 1.288 -o 5"
test8="RDDM -w 1 -o 5"
test9="RDDM -w 1 -o 3.5"
test10="RDDM -w 2.743 -o 3.5"
test11="RDDM -w 2.743 -o 5"
test12="RDDM -w 1.288 -o 2.258"

if [ $2 == "test0" ]; then
    TESTED_DETECTOR="RDDM -w 1.773 -o 2.258" # default
fi
if [ $2 == "test1" ]; then
    TESTED_DETECTOR="RDDM -w 2.258 -o 2.743"
fi
if [ $2 == "test2" ]; then
    TESTED_DETECTOR="RDDM -w 2.258 -o 5"
fi
if [ $2 == "test3" ]; then
    TESTED_DETECTOR="RDDM -w 2.258 -o 3.5"
fi
if [ $2 == "test4" ]; then
    TESTED_DETECTOR="RDDM -w 1.288 -o 1.773"
fi
if [ $2 == "test5" ]; then
    TESTED_DETECTOR="RDDM -w 1.288 -o 5"
fi
if [ $2 == "test6" ]; then
    TESTED_DETECTOR="RDDM -w 1.288 -o 3.5"
fi
if [ $2 == "test7" ]; then
    TESTED_DETECTOR="RDDM -w 1.288 -o 5"
fi
if [ $2 == "test8" ]; then
    TESTED_DETECTOR="RDDM -w 1 -o 5"
fi
if [ $2 == "test9" ]; then
    TESTED_DETECTOR="RDDM -w 1 -o 3.5"
fi
if [ $2 == "test10" ]; then
    TESTED_DETECTOR="RDDM -w 2.743 -o 3.5"
fi
if [ $2 == "test11" ]; then
    TESTED_DETECTOR="RDDM -w 2.743 -o 5"
fi
if [ $2 == "test12" ]; then
    TESTED_DETECTOR="RDDM -w 1.288 -o 2.258"
fi
if [ $2 == "test13" ]; then
    TESTED_DETECTOR="EDDM"
fi
if [ $2 == "test14" ]; then
    TESTED_DETECTOR="HDDM_A_Test"
fi
if [ $2 == "test15" ]; then
    TESTED_DETECTOR="ADDM"
fi
if [ $2 == "test16" ]; then
    TESTED_DETECTOR="RDDM -w 1.588 -o 2.258"
fi
if [ $2 == "test17" ]; then
    TESTED_DETECTOR="RDDM -w 1.33 -o 2.0"
fi
if [ $2 == "test18" ]; then
    TESTED_DETECTOR="RDDM -w 1.53 -o 2.2"
fi
if [ $2 == "test19" ]; then
    TESTED_DETECTOR="RDDM -w 1.23 -o 2.2"
fi
if [ $1 == "base" ]; then
    ALGOTOK="HT_base_${MAX_INSTANCES}"
    ALGORITHM=$HT
    testTTWindowOthers

    ALGOTOK="NB_base_${MAX_INSTANCES}"
    ALGORITHM=$INB
    testTTWindowOthers
fi


if [ $1 == "CPF_final" ]; then
    # # CPF / ECPF
    # CPF_fadeModelss=("-f" "")
    # # CPF_ms="0.85 0.925 0.95 0.975 0.99 0.9"  
    # CPF_min_buffer_sizes="30 60 120 180 240"
    # # CPF_fs="2 5 15" # comparable to max group size x number of groups

    CPF_baseClassifier=$INB   # $INB
    CPF_baseClassifier_name="NB"   # NB

    # FINAL SETUP ACCORDING TO BEST COMBINATION IN PO OFR INB
    CPF_driftDet="RDDM"
    CPF_m="0.99"
    CPF_f="15"
    CPF_min_buffer_size="120"
    CPF_fadeModels="-f"

    # tRIGGERING
    driftDetection=${CPF_driftDet}
    CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
    ALGORITHM=${CPF}
    ALGOTOK="CPF_final_${MAX_INSTANCES}_${CPF_baseClassifier_name}_${CPF_fadeModels}_${CPF_f}_${CPF_m}_${CPF_min_buffer_size}_${CPF_driftDet}"
    testTTWindowOthers
fi

if [ $1 == "ECPF_final" ]; then
    # # CPF / ECPF
    CPF_baseClassifier=$INB   # $INB
    CPF_baseClassifier_name="NB"   # NB
    # CPF_fadeModelss=("-f" "")
    # # CPF_ms="0.85 0.925 0.95 0.975 0.99 0.9"  
    # CPF_min_buffer_sizes="30 60 120 180 240"
    # # CPF_fs="2 5 15" # comparable to max group size x number of groups

    # FINAL SETUP ACCORDING TO BEST COMBINATION IN PO OFR INB
    CPF_driftDet="RDDM"
    CPF_m="0.925"
    CPF_f="15"
    CPF_fadeModels="-f"

    # tRIGGERING
    driftDetection=${CPF_driftDet}
    ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
    ALGORITHM=${ECPF}
    ALGOTOK="ECPF_final_${MAX_INSTANCES}_${CPF_baseClassifier_name}_${CPF_fadeModels}_${CPF_f}_${CPF_m}_${CPF_driftDet}"
    testTTWindowOthers
fi


if [ $1 == "ECPF_final_HT" ]; then
    # # CPF / ECPF
    CPF_baseClassifier=$HT   # $INB
    CPF_baseClassifier_name="HT"   # NB
    # CPF_fadeModelss=("-f" "")
    # # CPF_ms="0.85 0.925 0.95 0.975 0.99 0.9"  
    # CPF_min_buffer_sizes="30 60 120 180 240"
    # # CPF_fs="2 5 15" # comparable to max group size x number of groups

    # FINAL SETUP ACCORDING TO BEST COMBINATION IN PO OFR INB
    CPF_driftDet="RDDM"
    CPF_m="0.99"
    CPF_f="15"
    CPF_fadeModels="-f"

    # tRIGGERING
    driftDetection=${CPF_driftDet}
    ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
    ALGORITHM=${ECPF}
    ALGOTOK="ECPF_final_${MAX_INSTANCES}_${CPF_baseClassifier_name}_${CPF_fadeModels}_${CPF_f}_${CPF_m}_${CPF_driftDet}"
    testTTWindowOthers
fi

if [ $1 == "CPF_final_HT" ]; then
    # # CPF / ECPF
    CPF_baseClassifier=$HT   # $INB
    CPF_baseClassifier_name="HT"   # NB
    # CPF_fadeModelss=("-f" "")
    # # CPF_ms="0.85 0.925 0.95 0.975 0.99 0.9"  
    # CPF_min_buffer_sizes="30 60 120 180 240"
    # # CPF_fs="2 5 15" # comparable to max group size x number of groups

    # FINAL SETUP ACCORDING TO BEST COMBINATION IN PO OFR INB
    CPF_driftDet="EDDM"
    CPF_m="0.95"
    CPF_f="15"
    CPF_min_buffer_size="120"
    CPF_fadeModels="-f"

    # tRIGGERING
    driftDetection=${CPF_driftDet}
    CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
    ALGORITHM=${CPF}
    ALGOTOK="CPF_final_${MAX_INSTANCES}_${CPF_baseClassifier_name}_${CPF_fadeModels}_${CPF_f}_${CPF_m}_${CPF_min_buffer_size}_${CPF_driftDet}"
    testTTWindowOthers
fi


## best result in PO for NB in DEVSET
# ------------
# synthetic_1607572321_devset_NA_HT_5_1_2_5_200_EDDM_-T_-B__5_200_-F___w1000.txt 
# ==============
# NA_groupMaxSize = 5
# subtop = 1
# NA_distanceThresh = 2
# gng_lambda = 5
# gng_maxAge = 200
# minInsertionThresholdOption = 200
# driftDetectionAlgo = EDDM
# trainEND
# ww = 5
if [ $1 == "NA_tests_NB_best_config_31012021" ]; then
    NA_baseClassifier_name="NB"
    NA_baseClassifier=$INB  
    # RDDM with ww20
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_ww20_mg7_${TIMESTAMP}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # EDDM with ww5
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_EDDM_31012021_ww5_mg7_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi

if [ $1 == "NA_tests_HT_best_config_31012021" ]; then
    # RDDM with ww20
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_ww20_mg7_${TIMESTAMP}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # EDDM with ww5
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_EDDM_31012021_ww5_mg7_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi
if [ $1 == "NA_tests_NB_best_config_31012021_test16" ]; then
    NA_baseClassifier_name="NB"
    NA_baseClassifier=$INB  
    # RDDM with ww20 and test 16
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_ww20_mg7_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi
# end experiments for best results
#####################


##############################
##############################
### experiments added on Feb 2021 after here


# if [ $1 == "NA_tests_HT_best_config_EDDM_10Feb" ]; then
#     # RDDM with ww20
#     # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_ww20_mg7_${TIMESTAMP}"
#     # ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
#     #     -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
#     #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
#     # testTTWindow
#     # EDDM with ww5
#     ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_EDDM_10022021_ww5_mg7_${TIMESTAMP}"
#     ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
#         -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
#             -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
#     testTTWindow
# fi

if [ $1 == "NA_tests_HT_best_config_10Feb_test12_ww20" ]; then
    # RDDM with ww20
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_ww20_mg7_${TIMESTAMP}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # RDDM with ww20
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_10022021_ww20_mg7_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 20 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"         
    testTTWindow
fi

if [ $1 == "tests_dist5_ww20_insertExamplesOnly_10Feb_test12_ww20" ]; then
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_10022021_ww20_notInsED_insExO_notMC_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Q -T -C 0.5 -F -B -n -t 2 -D 1 -I 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindowTerm
    testTTWindow 
fi

if [ $1 == "NA_tests_NB_best_config_31012021_new_notpretrain" ]; then
    NA_baseClassifier_name="NB"
    NA_baseClassifier=$INB  
    # RDDM with ww20
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_ww20_mg7_${TIMESTAMP}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # EDDM with ww5
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH2_l${lambda}_EDDM_31012021NEW_ww5_mg7_${TIMESTAMP}"
    ALGORITHM="meta.NAeddm -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi

if [ $1 == "NA_tests_HT_best_config_31012021_new_notpretrain" ]; then
    MAX_INSTANCES=10000
    # RDDM with ww20
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_ww20_mg7_${TIMESTAMP}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # EDDM with ww5
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH2_l${lambda}_EDDM_31012021NEW_ww5_mg7_${TIMESTAMP}"
    ALGORITHM="meta.NAeddm -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi