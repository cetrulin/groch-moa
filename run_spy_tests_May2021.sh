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

cd $MOA_DEV/experiments/scripts


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
PROBLEM=$5
#"synthetic3_seeds"
DATASET="spy_train"
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
MAX_INSTANCES=2000000 # 500000 # 2000000 1000000 350000

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
  # Paths
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
#==================================================
# START FINAL EXPERIMENTS
if [ $4 == "NB" ]; then
    NA_baseClassifier=$INB  # $INB
    NA_baseClassifier_name="NB"   # NB
    CPF_baseClassifier=$INB   # $INB
    CPF_baseClassifier_name="NB"   # NB

    if [ $1 == "NA" ]; then
        NA_driftDet="${6//_B_/ }"   # change xBx split str for spaces in drift detectors with actual space
        NA_dist=$7
        ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
        ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
        testTTWindow
    elif [ $1 == "CPF" ]; then
        CPF_driftDet="${6//_B_/ }"   # change xBx split str for spaces in drift detectors with actual space
        CPF_m=$7
        CPF_f=$8
        CPF_min_buffer_size=${9}
        CPF_fadeModels=${10}

        driftDetection=${CPF_driftDet}
        CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
        ALGORITHM=${CPF}
        ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
        testTTWindowOthers
    elif [ $1 == "ECPF" ]; then
        CPF_driftDet="${6//_B_/ }"   # change xBx split str for spaces in drift detectors with actual space
        CPF_m=$7
        CPF_f=$8
        CPF_fadeModels=$9

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
        NA_driftDet="${6//_B_/ }"   # change xBx split str for spaces in drift detectors with actual space
        NA_dist=$7
        ALGOTOK="NA-${NA_baseClassifier_name}_${SEED}"
        ALGORITHM="meta.NA -l ${NA_baseClassifier} -B -L -T -C 0.5 -F -t ${NA_dist} -D 1 -z 1 -w 20 -n -I 200 -e ./NA_events_log_${ALGOTOK}.txt \
            -M 10 -P (Least Used Model) -G Old -k ${subtop1} \
                -c (GNG -l ${lambda} -c 2147483647 -b) -o ${MAHABSET} -d (${NA_driftDet})"            
        testTTWindow
    elif [ $1 == "CPF" ]; then
        CPF_driftDet="${6//_B_/ }"   # change xBx split str for spaces in drift detectors with actual space
        CPF_m=$7
        CPF_f=$8
        CPF_min_buffer_size=${9}
        CPF_fadeModels=${10}

        driftDetection=${CPF_driftDet}
        CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
        ALGORITHM=${CPF}
        ALGOTOK="CPF_${CPF_baseClassifier_name}_${SEED}"
        testTTWindowOthers
    elif [ $1 == "ECPF" ]; then
        CPF_driftDet="${6//_B_/ }"   # change xBx split str for spaces in drift detectors with actual space
        CPF_m=$7
        CPF_f=$8
        CPF_fadeModels=$9
        driftDetection=${CPF_driftDet}
        ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
        echo $ECPF
        ALGORITHM=${ECPF}
        ALGOTOK="ECPF_${CPF_baseClassifier_name}_${SEED}"
        testTTWindowOthers
    fi
fi
