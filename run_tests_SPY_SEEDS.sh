#!/bin/bash
# MOA_DEV=/home/bin/moa/src/moa-2017.06-sources
WEKALOC=${MOA_DEV}/weka
MOALOC=${MOA_DEV}/moa
JAVAAGENT=${MOA_DEV}/sizeofag-1.0.0.jar
#EXTRAPATH="./lib/colt.jar:../moaAlgorithms/classes:../wekaAlgorithms/classes:../weka-3-7-4/lib/libsvm.jar:/home/alex/src/gpsc/bin"
#EXTRAPATH="./lib/colt.jar:../moaAlgorithms/classes:../wekaAlgorithms/classes:../weka-3-7-4/lib/libsvm.jar:../gpsc/bin:../RCARF-master/bin"
EXTRAPATH="${MOA_DEV}/lib/*":"${MOA_DEV}/moa_classes/classifiers"
RESULTPATH=${RESULTS_PATH}
DATAPATH=${DATA_PATH}"/real"
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

# Synthetic set 3
PROBLEM=$5  # "spy_seeds"
DATASET=$6  # "spy_train"
#DATASET="synthetic_1607572321_devset"

SEED=$3
DATASETPATH=${DATAPATH}/${PROBLEM}/${SEED}
# raw sets for mahalanobis
# maraw="maraw_" # ""
# MAHABSET="${DATASETPATH}/1_equities_spy20200103T1407_indicators_best.arff;${DATASETPATH}/2_fixed_pref_pff20200106T1302_indicators_best.arff;${DATASETPATH}/3_real_state_vnq20200106T1215_indicators_best.arff;${DATASETPATH}/4_inter_bonds_bwx20200106T1531_indicators_best.arff"
# MAHABSET="synthetic_1607572321_before_dev"
# generated states for mahalanobis
maraw=""
MAHABSET="${DATASETPATH}/spy_mahalanobis_state_1.arff;${DATASETPATH}/spy_mahalanobis_state_2.arff;${DATASETPATH}/spy_mahalanobis_state_3.arff"

# Instances and evaluation
WSIZE=1000
BATCHSIZE=10
# MAX_INSTANCES=1000000 # 2000000 1000000 350000
# distThresh=5  # 1 for subtop1 by default
# lambda=5
MAX_INSTANCES=500000 # 500000 # 2000000 1000000 350000

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
    -f ${BATCHSIZE} " \
   > ${RESULTFOLDER}/${DATASET}_${ALGOTOK}_w${WSIZE}.txt
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

if [ $1 == "DWM" ]; then
  # # DWM
  DWM_baseClassifier=${INB}
  DWM_periods="1 25 50" # 75 100"
  DWM_betas="0.5 0.75" # "0.25 0.5 0.75"
  DWM_thetas="0.005 0.01 0.025" # 0.05"
  DWM_maxExpertss="1 2"
  if [ $1 == "DWM" ]; then
    # for DWM_period in $DWM_periods; do
      DWM_period=$2
    #   DWM_theta=$6
      for DWM_beta in $DWM_betas; do
        for DWM_theta in $DWM_thetas; do
          for DWM_maxExperts in $DWM_maxExpertss; do
            DWM="meta.DynamicWeightedMajority -l ${DWM_baseClassifier} -p ${DWM_period} -b ${DWM_beta} -t ${DWM_theta} -e ${DWM_maxExperts}" # (1 tree)
            ALGORITHM=${DWM}
            ALGOTOK="DWM_iNB_${DWM_maxExperts}_${DWM_period}_${DWM_beta}_${DWM_theta}"
            testTTWindowOthers 
          done
        done
      done
    # done
  fi
fi

if [ $1 == "CPF_old" ]; then
    # # CPF / ECPF

    CPF_baseClassifier=$HT   # $INB
    CPF_baseClassifier_name="HT"   # NB
    CPF_fadeModelss=("-f" "")
    # CPF_ms="0.85 0.925 0.95 0.975 0.99 0.9"  
    CPF_min_buffer_sizes="30 60 120 180 240"
    CPF_fs="2 5 15" # comparable to max group size x number of groups
    CPF_driftDet="EDDM"
    CPF_m=$2

    for CPF_fadeModels in "${CPF_fadeModelss[@]}"; do
        CPF_fadeModels="-f"
        for CPF_f in $CPF_fs; do
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${CPF_fadeModels}_${CPF_f}_${CPF_m}_${CPF_driftDet}"
            testTTWindowOthers
            for CPF_min_buffer_size in $CPF_min_buffer_sizes; do
                CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
                ALGORITHM=${CPF}
                ALGOTOK="CPF_${CPF_baseClassifier_name}_${CPF_fadeModels}_${CPF_f}_${CPF_m}_${CPF_min_buffer_size}_${CPF_driftDet}"
                testTTWindowOthers
            done
        done
    done
fi


# 000
if [ $1 == "000" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_notTestBKG_notTrainActOnWarn"
    ALGORITHM="meta.NA -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_notTestBKG_notTrainActOnWarn_forceEarlyDrifts"
    ALGORITHM="meta.NA -F -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_notTestBKG_notTrainActOnWarn_forceEarlyDrifts_ww50"
    ALGORITHM="meta.NA -F -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    testTTWindow
fi

# 001
if [ $1 == "001" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_notTestBKG_TrainActOnWarn"
    ALGORITHM="meta.NA -T -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow

    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_notTestBKG_TrainActOnWarn_forceEarlyDrifts"
    ALGORITHM="meta.NA -T -F -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_notTestBKG_TrainActOnWarn_forceEarlyDrifts_ww50"
    ALGORITHM="meta.NA -T -F -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    testTTWindow
fi

# 010
if [ $1 == "010" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestBKG_notTrainActOnWarn"
    ALGORITHM="meta.NA -K -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestBKG_notTrainActOnWarn_forceEarlyDrifts"
    ALGORITHM="meta.NA -K -F -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestBKG_notTrainActOnWarn_forceEarlyDrifts_ww50"
    ALGORITHM="meta.NA -K -F -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    testTTWindow
fi

# 011
if [ $1 == "011" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestBKG_TrainActOnWarn"
    ALGORITHM="meta.NA -K -T -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestBKG_TrainActOnWarn_forceEarlyDrifts"
    ALGORITHM="meta.NA -K -T -F -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestBKG_TrainActOnWarn_forceEarlyDrifts_ww50"
    ALGORITHM="meta.NA -K -T -F -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    testTTWindow
fi

# 100
if [ $1 == "100" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_notTestBKG_notTrainActOnWarn"
    ALGORITHM="meta.NA -B -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_notTestBKG_notTrainActOnWarn_forceEarlyDrifts"
    ALGORITHM="meta.NA -B -F -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    testTTWindow
fi

# 101
if [ $1 == "101" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_notTestBKG_TrainActOnWarn"
    ALGORITHM="meta.NA -B -T -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_notTestBKG_TrainActOnWarn_forceEarlyDrifts"
    ALGORITHM="meta.NA -B -T -F -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    testTTWindow
fi

# 110
if [ $1 == "110" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestBKG_notTrainActOnWarn"
    ALGORITHM="meta.NA -B -K -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestBKG_notTrainActOnWarn_forceEarlyDrifts"
    ALGORITHM="meta.NA -B -K -F -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    testTTWindow
fi

# 111 
if [ $1 == "111" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestBKG_TrainActOnWarn"
    ALGORITHM="meta.NA -B -K -T -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestBKG_TrainActOnWarn_forceEarlyDrifts"
    ALGORITHM="meta.NA -B -K -T -F -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    testTTWindow
fi

if [ $1 == "021" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts"
    ALGORITHM="meta.NA -L -C 0.5 -T -F -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn"
    ALGORITHM="meta.NA -L -C 0.5 -T -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_ww50"
    ALGORITHM="meta.NA -L -C 0.5 -T -F -D 1 -w 50 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    testTTWindow
fi

if [ $1 == "020" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts"
    ALGORITHM="meta.NA -L -C 0.5 -F -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn"
    ALGORITHM="meta.NA -L -C 0.5 -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_ww50"
    ALGORITHM="meta.NA -L -C 0.5 -F -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    testTTWindow
fi

if [ $1 == "120" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts"
    ALGORITHM="meta.NA -B -L -C 0.5 -F -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn"
    ALGORITHM="meta.NA -B -L -C 0.5 -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_ww50"
    ALGORITHM="meta.NA -B -L -C 0.5 -F -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    testTTWindow
fi

if [ $1 == "121" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts"
    ALGORITHM="meta.NA -B -L -C 0.5 -T -F -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn"
    ALGORITHM="meta.NA -B -L -C 0.5 -T -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_ww50"
    ALGORITHM="meta.NA -B -L -C 0.5 -T -F -D 1 -w 50 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    testTTWindow
fi

if [ $1 == "021_sub2" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_subtop2"
    ALGORITHM="meta.NA -L -C 0.5 -T -F -D 1 -z 1 -t 1.5 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_subtop2"
    ALGORITHM="meta.NA -L -C 0.5 -T -D 1 -z 1 -t 1.5 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_ww50_subtop2"
    ALGORITHM="meta.NA -L -C 0.5 -T -F -D 1 -w 50 -t 1.5 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    testTTWindow
fi

if [ $1 == "021_sub2_rddm" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_subtop2_rddm"
    ALGORITHM="meta.NA -L -C 0.5 -T -F -D 1 -z 1 -t 1.5 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d RDDM"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_subtop2_rddm"
    ALGORITHM="meta.NA -L -C 0.5 -T -D 1 -z 1 -t 1.5 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d RDDM"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_ww50_subtop2_rdmm_preCH"
    ALGORITHM="meta.NA -L -C 0.5 -T -F -n -D 1 -w 50 -t 1.5 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d RDDM"
    testTTWindow
fi

if [ $1 == "021_sub2_hddm" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_ww50_subtop2_hddma_preCH"
    ALGORITHM="meta.NA -L -C 0.5 -T -F -n -D 1 -w 50 -t 1.5 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d HDDM_A_Test"
    testTTWindow
fi
if [ $1 == "021_sub2_addm" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_ww50_subtop2_addm_preCH"
    ALGORITHM="meta.NA -L -C 0.5 -T -F -n -D 1 -w 50 -t 1.5 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d ADDM"
    testTTWindow
fi


if [ $1 == "021_sub1_rddm" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_subtop1_rddm"
    ALGORITHM="meta.NA -L -C 0.5 -T -F -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d RDDM"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_subtop1_rddm"
    ALGORITHM="meta.NA -L -C 0.5 -T -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d RDDM"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_ww50_subtop1_rddm_preCH${distThresh}_l${lambda}"
    ALGORITHM="meta.NA -L -C 0.5 -T -F -n -t ${distThresh} -D 1 -w 50 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d RDDM"
    testTTWindow
fi

# if [ $1 == "021_sub1_hddm" ]; then
#     ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_ww50_subtop1_hddma_preCH${distThresh}_l${lambda}"
#     ALGORITHM="meta.NA -L -C 0.5 -T -n -t ${distThresh} -F -D 1 -w 50 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
#         -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
#             -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d HDDM_A_Test"
#     testTTWindow
# fi

# if [ $1 == "021_sub1_addm" ]; then
#     ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_ww50_subtop1_addm_preCH${distThresh}_l${lambda}"
#     ALGORITHM="meta.NA -L -C 0.5 -T -F -n -t ${distThresh} -D 1 -w 50 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
#         -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
#             -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ADDM"
#     testTTWindow
# fi


if [ $1 == "020_rddm" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts"
    ALGORITHM="meta.NA -L -C 0.5 -F -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn"
    ALGORITHM="meta.NA -L -C 0.5 -D 1 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d EDDM2"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_hddma"
    ALGORITHM="meta.NA -L -C 0.5 -F -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 10 -c 2147483647 -b) -o ${MAHABSET} -d HDDM_A_Test"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_addm_preCH${distThresh}_l${lambda}"
    ALGORITHM="meta.NA -L -C 0.5 -F -n -t ${distThresh} -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ADDM"
    testTTWindow
fi

if [ $1 == "rddm" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_rddm_preCH${distThresh}_l${lambda}"
    ALGORITHM="meta.NA -L -C 0.5 -F -n -t ${distThresh} -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d RDDM"
    testTTWindow
fi


if [ $1 == "021_sub1_eddm_preTrain" ]; then
    ALGORITHM="meta.NA -L -C 0.5 ${updateTops} -T -F -n -t ${distThresh} -D 1 -w 50 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d EDDM"
    testTTWindow
fi

if [ $1 == "021_sub1_eddm_notPreTrain" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_ww50_subtop1_eddm_notPreCH${distThresh}_l${lambda}"
    ALGORITHM="meta.NA -L -C 0.5 ${updateTops} -T -F -t ${distThresh} -D 1 -w 50 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d EDDM"
    testTTWindow
fi



########################

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



####
# top experiments

if [ $1 == "021_sub1_hddm" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_ww50_subtop1_hddma_preCH${distThresh}_l${lambda}${updateTops}"
    ALGORITHM="meta.NA -L -C 0.5 ${updateTops} -T -n -t ${distThresh} -F -D 1 -w 50 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d HDDM_A_Test"
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_ww50_subtop1_hddma_notPreCH${distThresh}_l${lambda}${updateTops}"
    ALGORITHM="meta.NA -L -C 0.5 ${updateTops} -T -t ${distThresh} -F -D 1 -w 50 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d HDDM_A_Test"
    testTTWindow
fi

if [ $1 == "021_sub1_addm" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_ww50_subtop1_addm_preCH${distThresh}_l${lambda}${updateTops}"
    ALGORITHM="meta.NA -L -C 0.5 ${updateTops} -T -F -n -t ${distThresh} -D 1 -w 50 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ADDM"
    # testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_ww50_subtop1_addm_notPreCH${distThresh}_l${lambda}${updateTops}"
    ALGORITHM="meta.NA -L -C 0.5 ${updateTops} -T -F -t ${distThresh} -D 1 -w 50 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ADDM"
    testTTWindow
fi

if [ $1 == "021_sub1_eddm" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_ww50_subtop1_eddm_preCH${distThresh}_l${lambda}${updateTops}"
    ALGORITHM="meta.NA -L -C 0.5 ${updateTops} -T -F -n -t ${distThresh} -D 1 -w 50 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d EDDM"
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_ww50_subtop1_eddm_notPreCH${distThresh}_l${lambda}${updateTops}"
    ALGORITHM="meta.NA -L -C 0.5 ${updateTops} -T -F -t ${distThresh} -D 1 -w 50 -z 1 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d EDDM"
    testTTWindow
fi

# Last set of experiments

if [ $1 == "rddm_tests2_dist" ]; then
    test2="RDDM -w 2.258 -o 5"
    TEST=${test2}
    TEST_NAME="test2"

    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_rddm_notPreCH1_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -L -T -C 0.5 -F ${updateTops} -t 1 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow

    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_rddm_notPreCH5_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -L -T -C 0.5 -F ${updateTops} -t 5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi

if [ $1 == "rddm_tests2_dist_preTrain" ]; then
    test2="RDDM -w 2.258 -o 5"
    TEST=${test2}
    TEST_NAME="test2"

    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_rddm_preCH1_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -L -T -C 0.5 -F ${updateTops} -t 1 -n -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow

    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_rddm_preCH5_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -L -T -C 0.5 -F ${updateTops} -t 5 -n -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi

if [ $1 == "rddm_tests9_dist" ]; then
    TEST=${test9}
    TEST_NAME="test9"

    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_rddm_notPreCH1_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -L -T -C 0.5 -F ${updateTops} -t 1 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow

    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_rddm_notPreCH5_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -L -T  -C 0.5 -F ${updateTops} -t 5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi

if [ $1 == "rddm_tests9_dist_preTrain" ]; then
    TEST=${test12}
    TEST_NAME="test9"
    
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_rddm_preCH1_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -L -T -C 0.5 -F ${updateTops} -t 1 -n -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow

    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_rddm_preCH5_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -L -T -C 0.5 -F ${updateTops} -t 5 -n -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi

if [ $1 == "rddm_tests12_dist_preTrain" ]; then
    TEST=${test12}
    TEST_NAME="test12"

    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_rddm_preCH1_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -L -T -C 0.5 -F ${updateTops} -t 1 -n -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow

    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_rddm_preCH5_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -L -T -C 0.5 -F ${updateTops} -t 5 -n -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi

if [ $1 == "rddm_tests12_dist" ]; then
    TEST=${test12}
    TEST_NAME="test12"

    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_rddm_notPreCH1_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -L -T -C 0.5 -F ${updateTops} -t 1 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow

    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_rddm_notPreCH5_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -L -T -C 0.5 -F ${updateTops} -t 5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi


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

# lets not update topologies by now..
updateTops=""
if [ $1 == "NA_final_1" ]; then
    TEST_NAME=$1
    TEST="ADDM"
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_${TEST}_notPreCH150_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_28122020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F ${updateTops} -t 1.50 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    echo $ALGORITHM
    testTTWindow
fi

if [ $1 == "NA_final_1_old_warn_evaluation" ]; then
    # This disables -L so during warning only the active is considered for the evaluation
    # also this does not force early drifts
    TEST_NAME=$1
    TEST="ADDM"
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_${TEST}_notPreCH150_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_activeEvOnWW_28122020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -C 0.5 ${updateTops} -t 1.50 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    echo $ALGORITHM
    testTTWindow
fi

if [ $1 == "NA_final_2_old_warn_evaluation" ]; then
    # This disables -L so during warning only the active is considered for the evaluation
    # also this does not force early drifts
    TEST_NAME=$1
    TEST="ADDM"
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_${TEST}_notPreCH150_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_activeEvOnWW_28122020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -C 0.5 ${updateTops} -T -t 1.50 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    echo $ALGORITHM
    testTTWindow
fi

if [ $1 == "NA_final_2" ]; then
    TEST_NAME=$1
    TEST="ADDM"
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_${TEST}_notPreCH150_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_28122020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -T  ${updateTops} -t 1.50 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    echo $ALGORITHM
    testTTWindow
fi

if [ $1 == "NA_final_3" ]; then
    TEST_NAME=$1
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_RDDM_notPreCH150_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_29122020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F ${updateTops} -t 1.50 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    echo $ALGORITHM
    testTTWindow
fi

if [ $1 == "NA_final_4" ]; then
    TEST_NAME=$1
    # TEST="RDDM -o 2.0" 
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_notPreCH150_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_29122020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -T  ${updateTops} -t 1.50 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    echo $ALGORITHM
    testTTWindow
fi


if [ $1 == "NA_final_1_041012020" ]; then
    TEST_NAME=$1
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA1-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_RDDM_notPreCH1_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F ${updateTops} -t 1.0 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    echo $ALGORITHM
    testTTWindow
fi

if [ $1 == "NA_final_2_041012020" ]; then
    TEST_NAME=$1
    # TEST="RDDM -o 2.0" 
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA1-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_notPreCH1_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -T  ${updateTops} -t 1.0 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    echo $ALGORITHM
    testTTWindow
fi

if [ $1 == "NA_final_3_041012020" ]; then
    TEST_NAME=$1
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_RDDM_notPreCH2_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F ${updateTops} -t 2.0 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    echo $ALGORITHM
    testTTWindow
fi

if [ $1 == "NA_final_4_041012020" ]; then
    TEST_NAME=$1
    # TEST="RDDM -o 2.0" 
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_notPreCH2_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -T  ${updateTops} -t 2.0 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    echo $ALGORITHM
    testTTWindow
fi

if [ $1 == "NA_final_5_041012020" ]; then
    TEST_NAME=$1
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA2E-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_RDDM_notPreCH1_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -s Euclidean -L -C 0.5 -F ${updateTops} -t 1.0 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    echo $ALGORITHM
    testTTWindow
fi

if [ $1 == "NA_final_6_041012020" ]; then
    TEST_NAME=$1
    # TEST="RDDM -o 2.0" 
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA2E-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_notPreCH1_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -s Euclidean -L -C 0.5 -F -T  ${updateTops} -t 1.0 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    echo $ALGORITHM
    testTTWindow
fi

if [ $1 == "NA_final_1_pre_041012020" ]; then
    TEST_NAME=$1
    maxAge="200"
    wwMinThresh="20"
    # ALGOTOK="${maraw}NA1-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_RDDM_preCH2_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F ${updateTops} -n -t 2.0 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    # echo $ALGORITHM
    # testTTWindow
    NA_baseClassifier=$INB  # $INB
    NA_baseClassifier_name="INB"   # NB
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    testTTWindow
fi

if [ $1 == "NA_final_2_pre_041012020" ]; then
    TEST_NAME=$1
    # TEST="RDDM -o 2.0" 
    maxAge="200"
    wwMinThresh="20"
    # ALGOTOK="${maraw}NA1-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH2_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -T  ${updateTops} -n -t 2.0 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    # echo $ALGORITHM
    # testTTWindow
    NA_baseClassifier=$INB  # $INB
    NA_baseClassifier_name="INB"   # NB                            
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    testTTWindow
fi

if [ $1 == "NA_final_1_pre_041012020_END" ]; then
    TEST_NAME=$1
    maxAge="200"
    wwMinThresh="20"
    # ALGOTOK="${maraw}NA1-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_RDDM_preCH2_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -C 0.5 -F ${updateTops} -n -t 2.0 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    # echo $ALGORITHM
    # testTTWindow
    NA_baseClassifier=$INB  # $INB
    NA_baseClassifier_name="INB"   # NB
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -C 0.5 -F ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    testTTWindow
fi

if [ $1 == "NA_final_2_pre_041012020_END" ]; then
    TEST_NAME=$1
    # TEST="RDDM -o 2.0" 
    maxAge="200"
    wwMinThresh="20"
    # ALGOTOK="${maraw}NA1-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH2_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -C 0.5 -F -T  ${updateTops} -n -t 2.0 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    # echo $ALGORITHM
    # testTTWindow
    NA_baseClassifier=$INB  # $INB
    NA_baseClassifier_name="INB"   # NB
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -C 0.5 -F ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    testTTWindow
fi


if [ $1 == "NA_final_3_pre_041012020" ]; then
    TEST_NAME=$1
    maxAge="200"
    wwMinThresh="20"
    # ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
    #         -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    # echo $ALGORITHM
    # testTTWindow
    NA_baseClassifier=$INB  # $INB
    NA_baseClassifier_name="INB"   # NB
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    testTTWindow
fi

if [ $1 == "NA_final_4_pre_041012020" ]; then
    TEST_NAME=$1
    # TEST="RDDM -o 2.0" 
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -T  ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    echo $ALGORITHM
    testTTWindow
    # NA_baseClassifier=$INB  # $INB
    # NA_baseClassifier_name="INB"   # NB
    # ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -T  ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
    #         -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    # echo $ALGORITHM
    # testTTWindow
fi

if [ $1 == "NA_final_4_pre_041012020_END" ]; then
    TEST_NAME=$1
    # TEST="RDDM -o 2.0" 
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -C 0.5 -F -T  ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    echo $ALGORITHM
    testTTWindow
    # NA_baseClassifier=$INB  # $INB
    # NA_baseClassifier_name="INB"   # NB
    # ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -C 0.5 -F -T  ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
    #         -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    # echo $ALGORITHM
    # testTTWindow
fi


if [ $1 == "NA_final_4_pre_041012020_notforce" ]; then
    TEST_NAME=$1
    # TEST="RDDM -o 2.0" 
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -T  ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    echo $ALGORITHM
    testTTWindow
    # NA_baseClassifier=$INB  # $INB
    # NA_baseClassifier_name="INB"   # NB
    # ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -T  ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
    #         -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    # echo $ALGORITHM
    # testTTWindow
fi

if [ $1 == "NA_final_4_pre_041012020_notforce_" ]; then
    TEST_NAME=$1
    # TEST="RDDM -o 2.0" 
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -T  ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    echo $ALGORITHM
    testTTWindow
    # NA_baseClassifier=$INB  # $INB
    # NA_baseClassifier_name="INB"   # NB
    # ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -T  ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
    #         -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    # echo $ALGORITHM
    # testTTWindow
fi

if [ $1 == "NA_final_4_pre_041012020_notforce_notTestWeighted" ]; then
    TEST_NAME=$1
    # TEST="RDDM -o 2.0" 
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -C 0.5 -T  ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    echo $ALGORITHM
    testTTWindow
    # NA_baseClassifier=$INB  # $INB
    # NA_baseClassifier_name="INB"   # NB
    # ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -T  ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
    #         -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    # echo $ALGORITHM
    # testTTWindow
fi


if [ $1 == "NA_final_4_pre_041012020_notTestWeighted" ]; then
    TEST_NAME=$1
    # TEST="RDDM -o 2.0" 
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -F -C 0.5 -T  ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    echo $ALGORITHM
    testTTWindow
    # NA_baseClassifier=$INB  # $INB
    # NA_baseClassifier_name="INB"   # NB
    # ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -T  ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
    #         -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    # echo $ALGORITHM
    # testTTWindow
fi



if [ $1 == "NA_final_4_pre_041012020_END_notforce" ]; then
    TEST_NAME=$1
    # TEST="RDDM -o 2.0" 
    maxAge="200"
    wwMinThresh="20"
    ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -C 0.5 -T  ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    echo $ALGORITHM
    testTTWindow
    # NA_baseClassifier=$INB  # $INB
    # NA_baseClassifier_name="INB"   # NB
    # ALGOTOK="${maraw}NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_RDDM_preCH250_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -C 0.5 -F -T  ${updateTops} -n -t 2.5 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
    #         -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    # echo $ALGORITHM
    # testTTWindow
fi



if [ $1 == "NA_final_5_pre_041012020" ]; then
    TEST_NAME=$1
    maxAge="200"
    wwMinThresh="20"
    # ALGOTOK="${maraw}NA2E-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_RDDM_preCH1_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -s Euclidean -L -C 0.5 -F ${updateTops} -n -t 1.0 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
    #         -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    # echo $ALGORITHM
    # testTTWindow
    NA_baseClassifier=$INB  # $INB
    NA_baseClassifier_name="INB"   # NB
    ALGOTOK="${maraw}NA2E-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_RDDM_preCH1_l${lambda}_${TEST_NAME}${updateTops}_${maxAge}_${wwMinThresh}_final_041012020"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -s Euclidean -n -L -C 0.5 -F ${updateTops} -n -t 1.0 -D 1 -z 1 -w ${wwMinThresh} -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -m ${maxAge} -c 2147483647 -b) -o ${MAHABSET} -d (RDDM -o 2.0)"
    echo $ALGORITHM
    testTTWindow
fi

if [ $1 == "tests_dist_online" ]; then
    TEST=$TESTED_DETECTOR
    TEST_NAME=$2
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_TESTDETECTOR_notPreCH1_l${lambda}_${TEST_NAME}${updateTops}_online_1"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -H -F ${updateTops} -t 1 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
    $2="test12"
    TEST_NAME="test12"
    TESTED_DETECTOR="RDDM -w 1.288 -o 2.258"
    TEST=$TESTED_DETECTOR
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_TESTDETECTOR_preCH1_l${lambda}_${TEST_NAME}${updateTops}_online_1"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -H -C 0.5 -F ${updateTops} -n -t 1 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow

fi

if [ $1 == "tests_dist" ]; then
    TEST=$TESTED_DETECTOR
    TEST_NAME=$2

    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_${TEST}_notPreCH1_l${lambda}_${TEST_NAME}${updateTops}_notonline_1"
    ALGORITHM="meta.NA -L -C 0.5 -F ${updateTops} -t 1 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow

    # ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_${TEST}_notPreCH5_l${lambda}_${TEST_NAME}${updateTops}"
    # ALGORITHM="meta.NA -L -C 0.5 -F ${updateTops} -t 5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    # testTTWindow

    # ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_${TEST}_notPreCH1_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -L -C 0.5 -F -t 1 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    # testTTWindow

    # ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_${TEST}_notPreCH5_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -L -C 0.5 -F -t 5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    # testTTWindow

    # $2="test12"
    # TEST_NAME="test12"
    # TESTED_DETECTOR="RDDM -w 1.288 -o 2.258"
    # TEST=$TESTED_DETECTOR
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_TESTDETECTOR_preCH1_l${lambda}_${TEST_NAME}${updateTops}_notonline_1"
    ALGORITHM="meta.NA -L -C 0.5 -F ${updateTops} -n -t 1 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow

    # ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_${TEST}_preCH5_l${lambda}_${TEST_NAME}${updateTops}"
    # ALGORITHM="meta.NA -L -C 0.5 -F ${updateTops} -n -t 5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    # testTTWindow

    # ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_${TEST}_preCH1_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -L -C 0.5 -F -t 1 -n -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    # testTTWindow

    # ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_${TEST}_preCH5_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -L -C 0.5 -F -t 5 -n -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    # testTTWindow
fi


TEST=$TESTED_DETECTOR
TEST_NAME=$2
if [ $1 == "tests_dist2" ]; then
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F ${updateTops} -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
    ALGOTOK="NA${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F ${updateTops} -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi


updateTops="-j"
if [ $1 == "tests_dist4" ]; then
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}${updateTops}_08012021_T4"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F ${updateTops} -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}${updateTops}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F ${updateTops} -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
fi
if [ $1 == "tests_dist3" ]; then
#    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1_l${lambda}_${TEST_NAME}"
#     ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -t 1.0 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
#         -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
#             -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
#     testTTWindow
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1_l${lambda}_${TEST_NAME}_08012021_T3"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -t 1.0 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi

if [ $1 == "tests_dist5" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_08012021_T5"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_maxgroup1" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_08012021_T5_mg1"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 1 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_maxgroup3" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_08012021_T5_mg3"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 3 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_ww20" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_08012021_T5_ww20"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 3 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi


if [ $1 == "tests_dist5_ww20_TEST_ALIGNMENT" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_08012021_T5_ww20"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 3 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindowTerm
    testTTWindow
fi

if [ $1 == "tests_dist5_ww100" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_08012021_T5_ww100"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 100 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 3 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_ww20_mc" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_08mc2021_T5_ww20"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -Y -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 3 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_mc_1.5" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH1.5_l${lambda}_${TEST_NAME}_mc_T5_ww20"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -Y -L -T -C 0.5 -F -n -t 1.5 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_mc_2" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_mc_T5_ww20"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -Y -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_mc_2.5" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2.5_l${lambda}_${TEST_NAME}_mc_T5_ww20"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -Y -L -T -C 0.5 -F -n -t 2.5 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_not_mc_1.5" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH1.5_l${lambda}_${TEST_NAME}_not_mc_T5_ww20"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 1.5 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_not_mc_2" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_not_mc_T5_ww20"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_not_mc_2.5" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2.5_l${lambda}_${TEST_NAME}_not_mc_T5_ww20"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2.5 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_ww20" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_08132021_T5_ww20"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_ww50" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_08132021_T5_ww50"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 3 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_ww100_test" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_13012021_T5_ww100"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 100 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 3 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_ww100_t2" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_13mc2021_T5_ww100_t3"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -Y -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 100 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 3 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_ww100_test_mc" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_1201mc21_T5_ww100"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Y -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 100 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 3 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi


if [ $1 == "tests_dist4_NA2" ]; then
    ALGOTOK="NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}${updateTops}_08012021_T4"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F ${updateTops} -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}${updateTops}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F ${updateTops} -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
fi


if [ $1 == "tests_dist3_NA2" ]; then
#    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1_l${lambda}_${TEST_NAME}"
#     ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -t 1.0 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
#         -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
#             -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
#     testTTWindow
    ALGOTOK="NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1_l${lambda}_${TEST_NAME}_08012021_T3"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi


if [ $1 == "tests_dist5_NA2" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    ALGOTOK="NA2-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_08012021_T5"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop2} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi


updateTops=""

if [ $1 == "tests_dist2_END" ]; then
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -B -C 0.5 -F ${updateTops} -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -B -T -C 0.5 -F ${updateTops} -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi
if [ $1 == "tests_dist4_END" ]; then
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -B -C 0.5 -F ${updateTops} -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}${updateTops}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -B -T -C 0.5 -F ${updateTops} -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi
if [ $1 == "tests_dist3_END" ]; then
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1_l${lambda}_${TEST_NAME}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -B -C 0.5 -F -t 1.0 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1_l${lambda}_${TEST_NAME}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -B -T -C 0.5 -F -t 1.0 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi
if [ $1 == "tests_dist5_END" ]; then
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -B -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -B -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi













if [ $1 == "tests_dist6" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}"
    ALGORITHM="meta.NA -L -T -C 0.5 -F -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi
if [ $1 == "tests_dist7" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}"
    ALGORITHM="meta.NA -L -C 0.5 -F -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi
if [ $1 == "tests_dist8" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    ALGORITHM="meta.NA -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi
if [ $1 == "tests_dist9" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    ALGORITHM="meta.NA -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi
if [ $1 == "tests_dist10" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}"
    ALGORITHM="meta.NA -L -B -T -C 0.5 -F -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi
if [ $1 == "tests_dist11" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}"
    ALGORITHM="meta.NA -L -B -C 0.5 -F -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi
if [ $1 == "tests_dist12" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    ALGORITHM="meta.NA -L -B -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi
if [ $1 == "tests_dist13" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    ALGORITHM="meta.NA -L -B -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist14_1" ]; then
    MAX_INSTANCES=500000
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_notforceEarlyDrifts_preCH2_l${lambda}_TEST13"
    ALGORITHM="meta.NA -L -T -C 0.5 -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
# fi
# if [ $1 == "tests_dist14_2" ]; then
    MAX_INSTANCES=500000
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_TEST13"
    ALGORITHM="meta.NA -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi
if [ $1 == "tests_dist14_2" ]; then
    MAX_INSTANCES=500000
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_notforceEarlyDrifts_preCH2_l${lambda}_TEST13"
    ALGORITHM="meta.NA -L -C 0.5 -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
# fi
# if [ $1 == "tests_dist14_4" ]; then
    MAX_INSTANCES=500000
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_TEST13"
    ALGORITHM="meta.NA -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi

if [ $1 == "test_ww_50" ]; then
    MAX_INSTANCES=600000
    test='withIF50'
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_TEST${test}"
    ALGORITHM="meta.NA -L -F -C 0.5 -T -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 30 -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi

if [ $1 == "test_ww_alwaysForce" ]; then
    MAX_INSTANCES=600000
    test='withoutIF'
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_TEST${test}"
    ALGORITHM="meta.NA -L -F -C 0.5 -T -Z -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 30 -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi

if [ $1 == "test_ww10" ]; then
    MAX_INSTANCES=600000
    test='withIF10'
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_TEST${test}"
    ALGORITHM="meta.NA -L -F -C 0.5 -t 1.5 -D 1 -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 30 -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi

if [ $1 == "test_ww5" ]; then
    MAX_INSTANCES=600000
    test='withIF5'
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_TEST${test}"
    ALGORITHM="meta.NA -L -F -C 0.5 -T -t 1.5 -D 1 -z 1 -w 5 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 30 -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi

if [ $1 == "test_ww15" ]; then
    MAX_INSTANCES=600000
    test='withIF15'
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_TEST${test}"
    ALGORITHM="meta.NA -L -F -C 0.5 -T -t 1.5 -D 1 -z 1 -w 15 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 30 -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi

if [ $1 == "test_ww20" ]; then
    MAX_INSTANCES=600000
    test='withIF20'
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_TEST${test}"
    ALGORITHM="meta.NA -L -F -C 0.5 -T -t 1.5 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 30 -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi

if [ $1 == "test_ww25" ]; then
    MAX_INSTANCES=600000
    test='withIF25'
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_TEST${test}"
    ALGORITHM="meta.NA -L -F -C 0.5 -t 1.5 -D 1 -z 1 -w 25 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 30 -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi

if [ $1 == "test_insertion_threshold_no" ]; then
    MAX_INSTANCES=1000000
    EXPERIMENT_PATH='test_insertion_examples_trained_threshold'
    RESULTFOLDER=$RESULTFOLDER'/'${EXPERIMENT_PATH}
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_TEST_INSTHRESH_NO"
    ALGORITHM="meta.NA -L -F -C 0.5 -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 5 -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi

if [ $1 == "test_insertion_threshold_no_pre" ]; then
    MAX_INSTANCES=1000000
    EXPERIMENT_PATH='test_insertion_examples_trained_threshold'
    RESULTFOLDER=$RESULTFOLDER'/'${EXPERIMENT_PATH}
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_TEST_INSTHRESH_NO"
    ALGORITHM="meta.NA -L -F -C 0.5 -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 5 -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi


if [ $1 == "test_insertion_threshold_yes_1" ]; then
    MAX_INSTANCES=1000000
    EXPERIMENT_PATH='test_insertion_examples_trained_threshold'
    RESULTFOLDER=$RESULTFOLDER'/'${EXPERIMENT_PATH}
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_TEST_INSTHRESH_200"
    ALGORITHM="meta.NA -L -F -C 0.5 -t 1.5 -D 1 -I 200 -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 5 -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi

if [ $1 == "test_insertion_threshold_yes_1_pre" ]; then
    MAX_INSTANCES=1000000
    EXPERIMENT_PATH='test_insertion_examples_trained_threshold'
    RESULTFOLDER=$RESULTFOLDER'/'${EXPERIMENT_PATH}
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_TEST_INSTHRESH_200"
    ALGORITHM="meta.NA -L -F -C 0.5 -t 2 -n -D 1 -I 200 -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 5 -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi

if [ $1 == "test_insertion_threshold_yes_2" ]; then
    MAX_INSTANCES=1000000
    EXPERIMENT_PATH='test_insertion_examples_trained_threshold'
    RESULTFOLDER=$RESULTFOLDER'/'${EXPERIMENT_PATH}
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_TEST_INSTHRESH_500"
    ALGORITHM="meta.NA -L -F -C 0.5 -t 1.5 -D 1 -I 500 -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 5 -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi

if [ $1 == "test_insertion_threshold_yes_2_pre" ]; then
    MAX_INSTANCES=1000000
    EXPERIMENT_PATH='test_insertion_examples_trained_threshold'
    RESULTFOLDER=$RESULTFOLDER'/'${EXPERIMENT_PATH}
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_TEST_INSTHRESH_500"
    ALGORITHM="meta.NA -L -F -C 0.5 -n -t 2 -D 1 -I 500 -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l 5 -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi

if [ $1 == "tests_dist6o" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_7500"
    ALGORITHM="meta.NA -L -T -O 7500 -C 0.5 -F -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_10000"
    ALGORITHM="meta.NA -L -T -O 10000 -C 0.5 -F -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi
if [ $1 == "tests_dist7o" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_7500"
    ALGORITHM="meta.NA -L -C 0.5 -O 7500 -F -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_10000"
    ALGORITHM="meta.NA -L -C 0.5 -O 10000 -F -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi
if [ $1 == "tests_dist8o" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_7500"
    ALGORITHM="meta.NA -L -T -C 0.5 -O 7500 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_10000"
    ALGORITHM="meta.NA -L -T -C 0.5 -O 10000 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi
if [ $1 == "tests_dist9o" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_7500"
    ALGORITHM="meta.NA -L -C 0.5 -F -O 7500 -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_10000"
    ALGORITHM="meta.NA -L -C 0.5 -F -O 10000 -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi
if [ $1 == "tests_dist10o" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_7500"
    ALGORITHM="meta.NA -L -B -T -C 0.5 -F -O 7500 -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_10000"
    ALGORITHM="meta.NA -L -B -T -C 0.5 -F -O 10000 -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi
if [ $1 == "tests_dist11o" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_7500"
    ALGORITHM="meta.NA -L -B -C 0.5 -F -O 7500 -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_10000"
    ALGORITHM="meta.NA -L -B -C 0.5 -F -O 10000 -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi
if [ $1 == "tests_dist12o" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_7500"
    ALGORITHM="meta.NA -L -B -T -C 0.5 -F -O 7500 -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_10000"
    ALGORITHM="meta.NA -L -B -T -C 0.5 -F -O 10000 -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi
if [ $1 == "tests_dist13o" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_7500"
    ALGORITHM="meta.NA -L -B -C 0.5 -F -n -O 7500 -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_10000"
    ALGORITHM="meta.NA -L -B -C 0.5 -F -n -O 10000 -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi


if [ $1 == "tests_dist6o2" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_N7500"
    ALGORITHM="meta.NA -L -T -N 7500 -C 0.5 -F -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_N10000"
    ALGORITHM="meta.NA -L -T -N 10000 -C 0.5 -F -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi
if [ $1 == "tests_dist7o2" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_N7500"
    ALGORITHM="meta.NA -L -C 0.5 -N 7500 -F -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_N10000"
    ALGORITHM="meta.NA -L -C 0.5 -N 10000 -F -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi
if [ $1 == "tests_dist8o2" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_N7500"
    ALGORITHM="meta.NA -L -T -C 0.5 -N 7500 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_N10000"
    ALGORITHM="meta.NA -L -T -C 0.5 -N 10000 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi
if [ $1 == "tests_dist9o2" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_N7500"
    ALGORITHM="meta.NA -L -C 0.5 -F -N 7500 -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_N10000"
    ALGORITHM="meta.NA -L -C 0.5 -F -N 10000 -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi
if [ $1 == "tests_dist10o2" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_N7500"
    ALGORITHM="meta.NA -L -B -T -C 0.5 -F -N 7500 -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_N10000"
    ALGORITHM="meta.NA -L -B -T -C 0.5 -F -N 10000 -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi
if [ $1 == "tests_dist11o2" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_N7500"
    ALGORITHM="meta.NA -L -B -C 0.5 -F -N 7500 -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_N10000"
    ALGORITHM="meta.NA -L -B -C 0.5 -F -N 10000 -t 1.5 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"
    testTTWindow
fi
if [ $1 == "tests_dist12o2" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_N7500"
    ALGORITHM="meta.NA -L -B -T -C 0.5 -F -N 7500 -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_N10000"
    ALGORITHM="meta.NA -L -B -T -C 0.5 -F -N 10000 -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi
if [ $1 == "tests_dist13o2" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_N7500"
    ALGORITHM="meta.NA -L -B -C 0.5 -F -n -N 7500 -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainEND_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_N10000"
    ALGORITHM="meta.NA -L -B -C 0.5 -F -n -N 10000 -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist9o3" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_N7500_ww10"
    ALGORITHM="meta.NA -L -C 0.5 -F -N 7500 -n -t 2 -D 1 -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi
if [ $1 == "tests_dist9o4" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_N10000_ww10"
    ALGORITHM="meta.NA -L -C 0.5 -F -N 10000 -n -t 2 -D 1 -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi
if [ $1 == "tests_dist9o5" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_N-DEFAULT_ww10"
    ALGORITHM="meta.NA -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi
if [ $1 == "tests_dist9o6" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_N5000_ww10"
    ALGORITHM="meta.NA -L -C 0.5 -F -N 5000 -n -t 2 -D 1 -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi
if [ $1 == "tests_dist9o7" ]; then
    ALGOTOK="NA_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_N2500_ww10"
    ALGORITHM="meta.NA -L -C 0.5 -F -N 2500 -n -t 2 -D 1 -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_ww20_notInsED" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_16012021_T5_ww20_notinsED"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindowTerm
    testTTWindow
fi
if [ $1 == "tests_dist5_ww20_insED" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_16012021_T5_ww20_insED"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -J -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindowTerm
    testTTWindow
fi

if [ $1 == "tests_dist5_ww20_insertExamplesOnly_insED" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -I 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_T5_ww20_insED_insExO"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -J -Q -T -C 0.5 -F -n -t 2 -D 1 -z 1 -I 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindowTerm
    testTTWindow
fi

if [ $1 == "tests_dist5_ww20_insertExamplesOnly_notInsED" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_26012021_T5_ww20_notInsED_insExO"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Q -T -C 0.5 -F -n -t 2 -D 1 -z 1 -I 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindowTerm
    testTTWindow
fi

# if [ $1 == "tests_dist5_ww20_insertExamplesOnly_insED_NB" ]; then
#     # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
#     # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -I 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
#     #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
#     #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
#     # testTTWindow
#     NA_baseClassifier_name='NB'
#     NA_baseClassifier=$INB
#     ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_16012021_T5_ww20_insED_insExO"
#     ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -J -Q -T -C 0.5 -F -n -t 2 -D 1 -I 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
#         -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
#             -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
#     # testTTWindowTerm
#     testTTWindow
# fi

# if [ $1 == "tests_dist5_ww20_insertExamplesOnly_notInsED_NB" ]; then
#     # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
#     # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -I 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
#     #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
#     #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
#     # testTTWindow
#     NA_baseClassifier_name='NB'
#     NA_baseClassifier=$INB
#     ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_16012021_T5_ww20_notInsED_insExO"
#     ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Q -T -C 0.5 -F -n -t 2 -I 1 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
#         -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
#             -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
#     # testTTWindowTerm
#     testTTWindow
# fi

# if [ $1 == "tests_dist5_ww20_insertExamplesOnly_insED_MC" ]; then
#     # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
#     # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -I 1 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
#     #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
#     #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
#     # testTTWindow
#     # NA_baseClassifier_name='NB'
#     # NA_baseClassifier=$INB
#     ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_16012021_T5_ww20_insED_insExO_mc"
#     ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Y -J -Q -T -C 0.5 -F -n -t 2 -D 1 -I 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
#         -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
#             -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
#     # testTTWindowTerm
#     testTTWindow
# fi

# if [ $1 == "tests_dist5_ww20_insertExamplesOnly_notInsED_MC" ]; then
#     # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
#     # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -I 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
#     #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
#     #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
#     # testTTWindow
#     # NA_baseClassifier_name='NB'
#     # NA_baseClassifier=$INB
#     ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_16012021_T5_ww20_notInsED_insExO_mc"
#     ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Y -J -Q -T -C 0.5 -F -n -t 2 -I 1 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
#         -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
#             -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
#     # testTTWindowTerm
#     testTTWindow
# fi

if [ $1 == "tests_ww20_insertExamplesOnly_insED_NB_mc" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -I 1 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_26012021_T5_ww20_insED_insExO_mc_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Y -J -Q -T -C 0.5 -F -n -t 2 -I 1 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindowTerm
    testTTWindow
fi

if [ $1 == "tests_ww20_insertExamplesOnly_notInsED_NB_mc" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_26012021_T5_ww20_notInsED_insExO_mc_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Y -Q -T -C 0.5 -F -n -I 1 -t 2 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindowTerm
    testTTWindow
fi


if [ $1 == "tests_dist5_not_mc_1_notPre" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # NA_baseClassifier_name='NB'
    # NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1_l${lambda}_${TEST_NAME}_not_mc_T5_ww20_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -t 1 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_NB_mg10_ww20_no_mc" ]; then
    NA_baseClassifier_name="NB"
    NA_baseClassifier=$INB
    
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_15012021_T5_mg10_ww20_no_mc_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi


if [ $1 == "tests_dist5_NB_mg7" ]; then
    NA_baseClassifier_name="NB"
    NA_baseClassifier=$INB
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_08012021_T5"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_26012021_T5_mg7_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi


if [ $1 == "tests_dist5_NB_mg7_ww20" ]; then
    NA_baseClassifier_name="NB"
    NA_baseClassifier=$INB
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_08012021_T5"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_26012021_T5_mg7_ww20_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_NB_mg7_ww20_notTrain" ]; then
    NA_baseClassifier_name="NB"
    NA_baseClassifier=$INB
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_08012021_T5"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_26012021_T5_mg7_ww20_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_NB_mg7_quick" ]; then
    NA_baseClassifier_name="NB"
    NA_baseClassifier=$INB
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_08012021_T5"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow    
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_mg7_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi


if [ $1 == "NA_1501_tests_dist5_NB_mg7_quick" ]; then
    NA_baseClassifier_name="NB"
    NA_baseClassifier=$INB
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_08012021_T5"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow    
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_before15012021_T5_mg7_QUICKTESTS_${TIMESTAMP}"
    ALGORITHM="meta.NA_before15012021 -l ${NA_baseClassifier} -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
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

if [ $1 == "tests_dist5_ww20_insertExamplesOnly_insED_NB" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -I 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_T5_ww20_insED_insExO_notMC_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -J -Q -T -C 0.5 -F -n -I 1 -t 2 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindowTerm
    testTTWindow
fi

if [ $1 == "tests_dist5_ww20_insertExamplesOnly_notInsED_NB" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -I 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_T5_ww20_notInsED_insExO_notMC_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Q -T -C 0.5 -F -n -t 2 -D 1 -I 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindowTerm
    testTTWindow
fi

if [ $1 == "tests_dist5_ww20_insertExamplesOnly_insED_NB_MC" ]; then
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_notTrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -C 0.5 -F -n -t 2 -D 1 -z 1 -w 50 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 5 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_T5_ww20_insED_insExO_MC_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Y -J -Q -T -C 0.5 -F -n -t 2 -I 1 -D 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindowTerm
    testTTWindow
fi

if [ $1 == "NA_tests_NB_best_config_31012021_test12_only" ]; then
    NA_baseClassifier_name="NB"
    NA_baseClassifier=$INB  
    # RDDM with ww20
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_ww20_mg7_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
    # EDDM with ww5
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_EDDM_31012021_ww5_mg7_${TIMESTAMP}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    # testTTWindow
fi


if [ $1 == "tests_dist5_ww20_insertExamplesOnly_notInsED_trainEND_NB" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_T5_ww20_notInsED_insExO_notMC_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Q -T -C 0.5 -F -B -n -t 2 -D 1 -I 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindowTerm
    testTTWindow
fi

if [ $1 == "tests_dist5_ww20_insertExamplesOnly_notInsED_trainEND" ]; then
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_T5_ww20_notInsED_insExO_notMC_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Q -T -C 0.5 -F -B -n -t 2 -D 1 -I 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindowTerm
    testTTWindow
fi

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

# if [ $1 == "tests_dist5_ww20_insertExamplesOnly_10Feb_EDDM_ww5" ]; then
#     ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_EDDM_10022021_ww5_notInsED_insExO_notMC_${TIMESTAMP}"
#     ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Q -T -C 0.5 -F -B -n -t 2 -D 1 -I 1 -z 1 -w 5 -e ./NA_events_log_${ALGOTOK}.txt \
#         -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
#             -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
#     # testTTWindowTerm
#     testTTWindow 
# fi

### NB tests
if [ $1 == "NA_tests_NB_best_config_10022021" ]; then
    NA_baseClassifier_name="NB"
    NA_baseClassifier=$INB  
    # RDDM with ww20
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_10022021_ww20_mg7_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 20 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "tests_insertExamplesOnly_notInsED_NB_10022021" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_10022021_w20_insExO_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Q -T -C 0.5 -F -B -n -t 2 -D 1 -I 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi


### tests NB after PO feb 2021

if [ $1 == "feb21_NB_w20_insExO" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1_l${lambda}_${TEST_NAME}_13022021_w20_insExO_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Q -T -C 0.5 -F -B -t 1 -D 1 -I 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "feb21_NB_w20_trainSTART_insExO" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1_l${lambda}_${TEST_NAME}_13022021_w20_insExO_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Q -T -C 0.5 -F -t 1 -D 1 -I 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "feb21_NB_w20_trainSTART_pretrain_insExO" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_13022021_w20_insExO_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Q -T -C 0.5 -F -t 2 -D 1 -I 1 -n -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "feb21_NB_w20_pretrain_insExO" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_13022021_w20_insExO_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Q -T -C 0.5 -F -B -t 2 -D 1 -I 1 -n -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "feb21_NB_w10" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1_l${lambda}_${TEST_NAME}_13022021_w10_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -B -t 1 -D 1 -I 1 -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "feb21_NB_w10_trainSTART" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1_l${lambda}_${TEST_NAME}_13022021_w10_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -t 1 -D 1 -I 1 -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "feb21_NB_w10_trainSTART_pretrain" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_13022021_w10_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -t 2 -D 1 -I 1 -n -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "feb21_NB_w10_pretrain" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_13022021_w10_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -B -t 2 -D 1 -I 1 -n -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "feb21_NB_w20_insExO_bestdistPO" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.75_l${lambda}_${TEST_NAME}_13022021_w20_insExO_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Q -T -C 0.5 -F -B -t 1.75 -D 1 -I 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "feb21_NB_w20_trainSTART_insExO_bestdistPO" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.75_l${lambda}_${TEST_NAME}_13022021_w20_insExO_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Q -T -C 0.5 -F -t 1.75 -D 1 -I 1 -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "feb21_NB_w20_trainSTART_pretrain_insExO_bestdistPO" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2.5_l${lambda}_${TEST_NAME}_13022021_w20_insExO_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Q -T -C 0.5 -F -t 2.5 -D 1 -I 1 -n -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "feb21_NB_w20_pretrain_insExO_bestdistPO" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2.5_l${lambda}_${TEST_NAME}_13022021_w20_insExO_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -Q -T -C 0.5 -F -B -t 2.5 -D 1 -I 1 -n -z 1 -w 20 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "feb21_NB_w10_bestdistPO" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_13022021_w10_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -B -t 1.5 -D 1 -I 1 -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "feb21_NB_w10_trainSTART_bestdistPO" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_notPreCH1.5_l${lambda}_${TEST_NAME}_13022021_w10_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -t 1.5 -D 1 -I 1 -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "feb21_NB_w10_trainSTART_pretrain_bestdistPO" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainSTART_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2.5_l${lambda}_${TEST_NAME}_13022021_w10_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -t 2.5 -D 1 -I 1 -n -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi

if [ $1 == "feb21_NB_w10_pretrain_bestdistPO" ]; then
    NA_baseClassifier_name='NB'
    NA_baseClassifier=$INB
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2.5_l${lambda}_${TEST_NAME}_13022021_w10_${TIMESTAMP}"
    ALGORITHM="meta.NA -l ${NA_baseClassifier} -L -T -C 0.5 -F -B -t 2.5 -D 1 -I 1 -n -z 1 -w 10 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    testTTWindow
fi


# ==========================


if [ $1 == "NA_tests_HT_best_config_EDDM_10Feb" ]; then
    # RDDM with ww20
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_ww20_mg7_${TIMESTAMP}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # EDDM with ww5
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_EDDM_14022021_ww5_mg7_${TIMESTAMP}"
    ALGORITHM="meta.NAeddm -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi

if [ $1 == "tests_dist5_ww20_insertExamplesOnly_10Feb_EDDM_ww5" ]; then
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_EDDM_14022021_ww5_notInsED_insExO_notMC_${TIMESTAMP}"
    ALGORITHM="meta.NAeddm -l ${NA_baseClassifier} -L -Q -T -C 0.5 -F -B -n -t 2 -D 1 -I 1 -z 1 -w 5 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    # testTTWindowTerm
    testTTWindow 
fi

if [ $1 == "NA_tests_NB_best_config_31012021_new" ]; then
    NA_baseClassifier_name="NB"
    NA_baseClassifier=$INB  
    # RDDM with ww20
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_ww20_mg7_${TIMESTAMP}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # EDDM with ww5
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_EDDM_31012021NEW_ww5_mg7_${TIMESTAMP}"
    ALGORITHM="meta.NAeddm -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
    testTTWindow
fi

if [ $1 == "NA_tests_HT_best_config_31012021_new" ]; then
    # RDDM with ww20
    # ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_${TEST_NAME}_31012021_ww20_mg7_${TIMESTAMP}"
    # ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
    #     -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
    #         -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d ($TEST)"            
    # testTTWindow
    # EDDM with ww5
    ALGOTOK="NA-${NA_baseClassifier_name}_${MAX_INSTANCES}-examples_trainEND_TestWeighted_TrainActOnWarn_forceEarlyDrifts_preCH2_l${lambda}_EDDM_31012021NEW_ww5_mg7_${TIMESTAMP}"
    ALGORITHM="meta.NAeddm -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -n -t 2 -D 1 -z 1 -w 5 -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
        -M 7 -P (Least Used Model) -G Old -k ${subtop1} \
            -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (EDDM)"            
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



#==================================================
# START FINAL EXPERIMENTS


if [ $SEED == "1" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.588"
            NA_dist="2.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.0 -w 1.33"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.588"
            NA_dist="1.75"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.925"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "2" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.0 -w 1.33"
            NA_dist="2.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="180"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.0 -w 1.33"
            CPF_m="0.95"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.588"
            NA_dist="1.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.975"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "3" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.2 -w 1.53"
            NA_dist="2.25"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.99"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.288"
            CPF_m="0.975"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 1.8 -w 1.13"
            NA_dist="2.25"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "4" ]; then
    MAX_INSTANCES=500000

    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.0 -w 1.33"
            NA_dist="1.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.975"
            CPF_f="4"
            CPF_min_buffer_size="180"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.925"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.2 -w 1.53"
            NA_dist="1.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.975"
            CPF_f="4"
            CPF_min_buffer_size="180"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "5" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.0 -w 1.33"
            NA_dist="1.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.975"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.588"
            NA_dist="2.25"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.975"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "6" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 1.8 -w 1.13"
            NA_dist="1.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.975"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 1.8 -w 1.13"
            NA_dist="1.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.975"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "7" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.588"
            NA_dist="2"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.925"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.2 -w 1.23"
            NA_dist="2"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.975"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.975"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "8" ]; then
    MAX_INSTANCES=500000

    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.288"
            NA_dist="1.75"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.0 -w 1.03"
            NA_dist="2.25"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.975"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.925"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "9" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.0 -w 1.03"
            NA_dist="2.25"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.588"
            NA_dist="1.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.99"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.975"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi


if [ $SEED == "10" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.0 -w 1.33"
            NA_dist="1.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="180"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.0 -w 1.33"
            CPF_m="0.975"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.2 -w 1.23"
            NA_dist="1.75"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.975"
            CPF_f="4"
            CPF_min_buffer_size="180"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.975"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "11" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.0 -w 1.33"
            NA_dist="1.75"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.58"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="180"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 1.8 -w 1.13"
            CPF_m="0.95"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.588"
            NA_dist="1.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="180"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "12" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.0 -w 1.03"
            NA_dist="1.75"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.0 -w 1.33"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="EDDM"
            NA_dist="1.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.975"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "13" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.288"
            NA_dist="2.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.975"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.2 -w 1.23"
            NA_dist="2.25"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.95"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "14" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.0 -w 1.33"
            NA_dist="1.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.99"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi

    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet='RDDM -o 2.258 -w 1.588'
            NA_dist="2.25"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.925"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "15" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.0 -w 1.03"
            NA_dist="1.75"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.0 -w 1.33"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.0 -w 1.33"
            NA_dist="2.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.975"
            CPF_f="4"
            CPF_min_buffer_size="180"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "16" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 1.8 -w 1.13"
            NA_dist="1.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.95"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.0 -w 1.33"
            NA_dist="1.75"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.95"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "17" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.2 -w 1.53"
            NA_dist="2"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.925"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.588"
            NA_dist="1.75"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.975"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "18" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.0 -w 1.33"
            NA_dist="2.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.975"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 1.8 -w 1.13"
            NA_dist="2"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.975"
            CPF_f="4"
            CPF_min_buffer_size="180"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.975"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "19" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.588"
            NA_dist="1.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.975"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.95"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.2 -w 1.53"
            NA_dist="2.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "20" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 1.8 -w 1.13"
            NA_dist="2"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.0 -w 1.33"
            CPF_m="0.925"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.588"
            NA_dist="1.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.975"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi

if [ $SEED == "21" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.2 -w 1.53"
            NA_dist="1.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.0 -w 1.33"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.588"
            NA_dist="2.25"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.95"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi


if [ $SEED == "22" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.2 -w 1.53"
            NA_dist="2"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.0 -w 1.33"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.0 -w 1.33"
            CPF_m="0.925"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.0 -w 1.03"
            NA_dist="2"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.95"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi


if [ $SEED == "23" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.2 -w 1.53"
            NA_dist="1.75"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="180"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.588"
            NA_dist="2"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi


if [ $SEED == "24" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.2 -w 1.23"
            NA_dist="1.75"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.975"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.95"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.0 -w 1.33"
            NA_dist="1.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.95"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi


if [ $SEED == "25" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.0 -w 1.33"
            NA_dist="1.75"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.975"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.588"
            NA_dist="1.75"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.975"
            CPF_f="4"
            CPF_min_buffer_size="180"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi


if [ $SEED == "26" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.2 -w 1.23"
            NA_dist="1.75"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.975"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.588"
            NA_dist="1.75"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.99"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi


if [ $SEED == "27" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 1.8 -w 1.13"
            NA_dist="1.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.0 -w 1.33"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.2 -w 1.53"
            NA_dist="2"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="180"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi


if [ $SEED == "28" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.0 -w 1.33"
            NA_dist="2"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="180"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.2 -w 1.23"
            NA_dist="2"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi


if [ $SEED == "29" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 1.8 -w 1.13"
            NA_dist="2.25"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.99"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.0 -w 1.33"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.588"
            NA_dist="2.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="180"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.99"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi


if [ $SEED == "30" ]; then
    MAX_INSTANCES=500000
    
    if [ $4 == "NB" ]; then
        NA_baseClassifier=$INB  # $INB
        NA_baseClassifier_name="NB"   # NB
        CPF_baseClassifier=$INB   # $INB
        CPF_baseClassifier_name="NB"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.288"
            NA_dist="2.5"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.925"
            CPF_f="4"
            CPF_min_buffer_size="60"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.2 -w 1.53"
            CPF_m="0.975"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    elif [ $4 == "HT" ]; then
        NA_baseClassifier=$HT  # $INB
        NA_baseClassifier_name="HT"   # NB
        CPF_baseClassifier=$HT   # $INB
        CPF_baseClassifier_name="HT"   # NB

        if [ $1 == "NA" ]; then
            NA_driftDet="RDDM -o 2.258 -w 1.588"
            NA_dist="1.75"
            ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
            ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
                -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                    -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
            testTTWindow
        elif [ $1 == "CPF" ]; then
            CPF_driftDet="EDDM"
            CPF_m="0.95"
            CPF_f="4"
            CPF_min_buffer_size="120"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
            ALGORITHM=${CPF}
            ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        elif [ $1 == "ECPF" ]; then
            CPF_driftDet="RDDM -o 2.258 -w 1.588"
            CPF_m="0.925"
            CPF_f="4"
            CPF_fadeModels=""
            driftDetection=${CPF_driftDet}
            ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
            ALGORITHM=${ECPF}
            ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
            testTTWindowOthers
        fi
    fi
fi


