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

# Datasets related
PROBLEMS=$3
DATASET=$4
mahalanobis_init_set=$5
MAX_INSTANCES=1000000

SEED="1"
WSIZE=1000 # 10 #500
ENSEMBLESIZE="1"

#SEEDS="1"
#SEEDS="3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

# Old
# INB="bayes.NaiveBayes"
# HOEFT="trees.HoeffdingTree"
# AHOEFT="trees.HoeffdingAdaptiveTree -g 10 -t 0.27 -l MC"

# SGD='meta.WEKAClassifier -l (weka.classifiers.functions.SGD)'
# DWM="meta.DynamicWeightedMajority -l (bayes.NaiveBayes)"
# DWMTREE="meta.DynamicWeightedMajority -l (trees.HoeffdingTree)"
# LNSENB='meta.LearnNSE -p 250  -l (meta.WEKAClassifier -w 250 -i 250 -f 250 -l (weka.classifiers.bayes.NaiveBayes))'
# LNSETREE='meta.LearnNSE -p 70 -l trees.HoeffdingTree'
# OZABAG='meta.OzaBagAdwin -l (bayes.NaiveBayes)'
# OZABAGTREE='meta.OzaBagAdwin -l (trees.HoeffdingTree)'
# OZABOOST='meta.OzaBoostAdwin -l (bayes.NaiveBayes)'
# OZABOOSTTREE='meta.OzaBoostAdwin -l (trees.HoeffdingTree)'
# ONSBOOST='meta.ONSBoost -C (bayes.NaiveBayes)'
# ONSBOOSTTREE='meta.ONSBoost -C (trees.HoeffdingTree)'
# # ARF='meta.AdaptiveRandomForest'
# # RCARF='meta.RecurringConceptsAdaptiveRandomForest'

# ARF='meta.AdaptiveRandomForest -s '$ENSEMBLESIZE' -l (ARFHoeffdingTree -k 5 -e 2000000 -g 50 -c 0.01) -x (ADWINChangeDetector -a 0.15) -p (ADWINChangeDetector -a 0.3)' 
# ARF_fast='meta.AdaptiveRandomForest -s '$ENSEMBLESIZE' -l (ARFHoeffdingTree -k 5 -e 2000000 -g 50 -c 0.01) -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01)' 
# ARF_moderate='meta.AdaptiveRandomForest -s '$ENSEMBLESIZE' -l (ARFHoeffdingTree -k 5 -e 2000000 -g 50 -c 0.01) -x (ADWINChangeDetector -a 0.00001) -p (ADWINChangeDetector -a 0.0001)' 
# #RCARF='meta.RecurringConceptsAdaptiveRandomForest -l (ARFHoeffdingTree -k 5 -e 2000000 -g 50 -c 0.01) -s 15 -x (ADWINChangeDetector -a 0.15) -p (ADWINChangeDetector -a 0.3) -e '${MOA_DEV}'/experiments/results/'${PROBLEM}'/1/'${PROBLEM}'_RCARF_EVENTS_w'$WSIZE'.txt'
# #RCARF_resize_all='meta.RecurringConceptsAdaptiveRandomForest -l (ARFHoeffdingTree -k 5 -e 2000000 -g 50 -c 0.01) -s 15 -x (ADWINChangeDetector -a 0.15) -p (ADWINChangeDetector -a 0.3) -b -e '${MOA_DEV}'/experiments/results/'${PROBLEM}'/1/'${PROBLEM}'RCARFall_EVENTS_'$WSIZE'.txt'
# #RCARF_fast='meta.RecurringConceptsAdaptiveRandomForest -l (ARFHoeffdingTree -k 5 -e 2000000 -g 50 -c 0.01) -s 15 -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01) -e '${MOA_DEV}'/experiments/results/'${PROBLEM}'/1/'${PROBLEM}'_RCARFfast_EVENTS_w'$WSIZE'.txt'
# #RCARF_fast_resize_all='meta.RecurringConceptsAdaptiveRandomForest -l (ARFHoeffdingTree -k 5 -e 2000000 -g 50 -c 0.01) -s 15 -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01) -b -e '${MOA_DEV}'/experiments/results/'${PROBLEM}'/1/'${PROBLEM}'_RCARFallfast_EVENTS_w'$WSIZE'.txt'
# #RCARF defined below

# RCARF_config2='meta.RecurringConceptsAdaptiveRandomForest -l (ARFHoeffdingTree -k 5 -e 2000000 -g 50 -c 0.01) -s 15 -x (ADWINChangeDetector -a 0.01) -p (ADWINChangeDetector -a 0.05)'
# RCARF_config2_all='meta.RecurringConceptsAdaptiveRandomForest -l (ARFHoeffdingTree -k 5 -e 2000000 -g 50 -c 0.01) -s 15 -x (ADWINChangeDetector -a 0.01) -p (ADWINChangeDetector -a 0.05) -b'
# RCARF_config3='meta.RecurringConceptsAdaptiveRandomForest -l (ARFHoeffdingTree -k 5 -e 2000000 -g 50 -c 0.01) -s 15 -x (ADWINChangeDetector -a 0.01) -p (ADWINChangeDetector -a 0.1)'
# RCARF_config3_all='meta.RecurringConceptsAdaptiveRandomForest -l (ARFHoeffdingTree -k 5 -e 2000000 -g 50 -c 0.01) -s 15 -x (ADWINChangeDetector -a 0.01) -p (ADWINChangeDetector -a 0.1) -b'

# RCD_HOEF="meta.RCD -t 10 -c 70 -l trees.HoeffdingTree -d (ADWINChangeDetector -a 0.15)"
# #"meta.RCD -l trees.HoeffdingTree"
# RCD_ARFHOEFT="meta.RCD -l trees.ARFHoeffdingTree"
# RCD_AHOEFT="meta.RCD -l trees.HoeffdingAdaptiveTree"
# NN100='meta.WEKAClassifier -l (weka.classifiers.lazy.IBk -K 1 -W 100 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"")'
# NN1500='meta.WEKAClassifier -l (weka.classifiers.lazy.IBk -K 1 -W 1500 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"")'
# NN6000='meta.WEKAClassifier -l (weka.classifiers.lazy.IBk -K 1 -W 6000 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"")'
# NN10000='meta.WEKAClassifier -l (weka.classifiers.lazy.IBk -K 1 -W 10000 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"")'

# ###############################
# ###############################

# Plain Vanilla Setup
# ###############

# Common params
warningDet="ADWINChangeDetector -a 1.0E-4"
driftDet="ADWINChangeDetector -a 1.0E-5" # optimise

# NA-specific params
feature_subset_indexes="1,3,12,15,16,17"
gng_lambda="50"
#gng_maxAge="200"
distanceTresh="1"
groupMaxSize="5"

# ###############################
# ###############################



# FOR THE ARTIFICIAL DATASETS THAT HAVE PROBABILITIES AS FINAL COLUMNS
# Strip probabilities from the mail file
#
makeTrainSet() {
  if [ -e ${TRAINDATA} ]; then
    rm  ${TRAINDATA}
  fi

  DIM=$(grep dimensions ${FULLDATA} | sed 's/.*: \([0-9]*\),/\1/')
  echo "Creating ${TRAINDATA} with dimension ${DIM}, last field is the class"
  awk -v f=1 -v t=${DIM} -f ./getcolumns.awk ${FULLDATA} > ${TRAINDATA}
}

makeTrainSet_COPY() {

  SEEDFOLDER=${DATAPATH}/${PROBLEM}/${SEED}
  if [ ! -d ${SEEDFOLDER} ]; then
    mkdir  ${SEEDFOLDER}
  fi

  TRAINDATA=${SEEDFOLDER}/${DATANAME}_train.arff
  FULLDATA=${DATAPATH}/${PROBLEM}/${DATANAME}.arff

  if [ -e ${TRAINDATA} ]; then
    rm  ${TRAINDATA}
  fi

  echo "Copying from ${FULLDATA} to ${TRAINDATA} last field is the class"
  cp ${FULLDATA} ${TRAINDATA}
}

#
# Test modes and evaluators
#
testTTold() {
  echo
  echo MOA12 Problem ${PROBLEM}, Running $ALGORITHM with aggregated evaluation
  echo Train data ${TRAINDATA}
  echo

  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_dump.txt
  # java -cp $MOALOC/moa.jar:$WEKALOC/weka.jar:$EXTRAPATH -javaagent:$JAVAAGENT moa.DoTask \
  java -cp $EXTRAPATH:$MOA_DEV -javaagent:$JAVAAGENT moa.DoTask \
  "EvaluateInterleavedTestThenTrain \
    -l (${ALGORITHM}) \
    -s (ArffFileStream -f ${TRAINDATA}) \
    -i ${MAX_INSTANCES} \
    -d ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_dump.txt \
    -r ${SEED} \
    -f 10 " \
  > ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt
  #  -O ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_task.txt \
  # zip "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.zip" "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt"
  # rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt
}

testTToldWindow() {
  echo
  echo MOA12 Problem ${PROBLEM}, Running $ALGORITHM with window ${WSIZE}
  echo Train data ${TRAINDATA}
  echo

  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}_dump.txt
  # java -cp $MOALOC/moa.jar:$WEKALOC/weka.jar:$EXTRAPATH -javaagent:$JAVAAGENT moa.DoTask \
  # -s (ArffFileStream -f ${TRAINDATA}) \ -e (WindowClassificationPerformanceEvaluator -w ${WSIZE}) \
  java -cp $EXTRAPATH:$MOA_DEV -javaagent:$JAVAAGENT moa.DoTask \
  "EvaluateInterleavedTestThenTrain \
    -l (${ALGORITHM}) \
    -s (ArffFileStream -f ${TRAINDATA}) \
    -i ${MAX_INSTANCES} \
    -d ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}_dump.txt \
    -r ${SEED} \
    -e (WindowClassificationPerformanceEvaluator -w ${WSIZE}) \
    -f 10 " \
  > ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.txt
  #   -O ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}_task.txt \
  #  zip "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.zip" "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.txt"
  #  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.txt
}

testTT() {
  echo
  echo Problem ${PROBLEM}, Running $ALGORITHM with aggregated evaluation
  echo Train data ${TRAINDATA}
  echo Test 1
  echo

  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_dump.txt
  # java -cp $MOALOC/moa.jar:$WEKALOC/weka.jar:$EXTRAPATH -javaagent:$JAVAAGENT moa.DoTask \
  # -l (${ALGORITHM} \
  #  -s 25 \
  #  -x (ADWINChangeDetector -a 0.13514) \
  #  -p (ADWINChangeDetector -a 0.49189)) \
  echo "%%%%%%%%%%%%%%%%%%%%"
  echo "java -cp $EXTRAPATH:$MOA_DEV -javaagent:$JAVAAGENT moa.DoTask \
  EvaluateInterleavedTestThenTrain \
  -l (${ALGORITHM}) \
  -s (ArffFileStream -f ${TRAINDATA} ) \
  -d ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_dump.txt \
  -r ${SEED} \
  -f 10 "
  echo "%%%%%%%%%%%%%%%%%%%%"

  java -cp $EXTRAPATH:$MOA_DEV -javaagent:$JAVAAGENT moa.DoTask \
  "EvaluateInterleavedTestThenTrain \
  -l (${ALGORITHM}) \
  -s (ArffFileStream -f ${TRAINDATA} ) \
  -i ${MAX_INSTANCES} \
  -d ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_dump.txt \
  -r ${SEED} \
  -f 10 " \
  > ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt
  # -O ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_task.txt \
  # zip "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.zip" "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt"
  # rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt
  # mv weka.log ${RESULTPATH}/${DATANAME}_${ALGOTOK}.log
}

testTTWindow() {
  echo
  echo Problem ${PROBLEM}, Running $ALGORITHM with window ${WSIZE}
  echo Train data ${TRAINDATA}
  echo Test 2
  echo

  # java -cp $MOALOC/moa.jar:$WEKALOC/weka.jar:$EXTRAPATH -javaagent:$JAVAAGENT moa.DoTask \
  #  -l (${ALGORITHM} \
  #  -s 25 \
  #  -x (ADWINChangeDetector -a 0.13514) \
  #  -p (ADWINChangeDetector -a 0.49189)) \
  #  -s (ArffFileStream -f ${TRAINDATA}) \
  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}_dump.txt
  java -cp $EXTRAPATH:$MOA_DEV -javaagent:$JAVAAGENT moa.DoTask \
  "EvaluateInterleavedTestThenTrain \
    -l (${ALGORITHM}) \
    -s (ArffFileStream -f ${TRAINDATA}) \
    -i ${MAX_INSTANCES} \
    -d ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}_dump.txt \
    -r ${SEED} \
    -e (WindowClassificationPerformanceEvaluator -w ${WSIZE}) \
    -f 10 " \
  > ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.txt
  # -O ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}_task.txt \
  #  zip "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.zip" "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.txt"
  #  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.txt
}

testTTC () {
  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt
  # java -cp $MOALOC/moa.jar:$WEKALOC/weka.jar:$EXTRAPATH -javaagent:$JAVAAGENT moa.DoTask \
  java -cp $EXTRAPATH:$MOA_DEV -javaagent:$JAVAAGENT moa.DoTask \
  "EvaluateInterleavedChunks \
    -l (${ALGORITHM}) \
    -s (ArffFileStream -f ${TRAINDATA}) \
    -i ${MAX_INSTANCES} \
    -d ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_dump.txt \
    -r ${SEED} \
    -c ${block_train} \
    -f ${block_test}" \
  > ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt
  #   -O ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_task.txt \
  #  zip "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.zip" "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt"
  #  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt
}

testTTCWindow () {
  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.txt
  # java -cp $MOALOC/moa.jar:$WEKALOC/weka.jar:$EXTRAPATH -javaagent:$JAVAAGENT moa.DoTask \
  java -cp $EXTRAPATH:$MOA_DEV -javaagent:$JAVAAGENT moa.DoTask \
  "EvaluateInterleavedChunks \
    -l (${ALGORITHM}) \
    -s (ArffFileStream -f ${TRAINDATA}) \
    -i ${MAX_INSTANCES} \
    -d ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_dump.txt \
    -r ${SEED} \
    -c ${block_train} \
    -e (WindowClassificationPerformanceEvaluator -w ${WSIZE}) \
    -f ${block_test}" \
  > ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.txt
  #    -O ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_task.txt \
  #  zip "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.zip" "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.txt"
  #  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.txt
}


testHT () {
  rm ${DATAPATH}/${PROBLEM}/${PROBLEM}_test.arff
  # java -cp $MOALOC/moa.jar:$WEKALOC/weka.jar:$EXTRAPATH -javaagent:$JAVAAGENT moa.DoTask \
  java -cp $EXTRAPATH:$MOA_DEV -javaagent:$JAVAAGENT moa.DoTask \
  "EvaluateChunksTwoFiles \
    -l (${ALGORITHM}) \
    -s (ArffFileStream -f ${TRAINDATA}) \
    -i ${MAX_INSTANCES} \
    -d ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_dump.txt \
    -r ${SEED} \
    -c ${block_train} \
    -y (ArffFileStream -f ${DATAPATH}/${PROBLEM}/${PROBLEM}_test.arff) \
    -z ${block_test} \
    -f ${block_test}" \
  > ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt
  #    -O ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_task.txt \
  #  zip "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.zip" "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt"
  #  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt
}

testHTWindow () {
  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}_dump.txt
  # java -cp $MOALOC/moa.jar:$WEKALOC/weka.jar:$EXTRAPATH -javaagent:$JAVAAGENT moa.DoTask \
  java -cp $EXTRAPATH:$MOA_DEV -javaagent:$JAVAAGENT moa.DoTask \
  "EvaluateChunksTwoFiles \
    -l (${ALGORITHM}) \
    -s (ArffFileStream -f ${TRAINDATA}) \
    -i ${MAX_INSTANCES} \
    -d ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}_dump.txt \
    -r ${SEED} \
    -c ${block_train} \
    -y (ArffFileStream -f ${DATAPATH}/${PROBLEM}/${PROBLEM}_test.arff) \
    -z ${block_test} \
    -e (WindowClassificationPerformanceEvaluator -w ${WSIZE}) \
    -f ${block_test}" \
  > ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.txt
  #   -O ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}_task.txt \
  #  zip "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.zip" "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.txt"
  #  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.txt
}

#
# MAIN LOOP
#
# Executes all the algorithms and dumps results to a file in the same
# data folder structure, under RESULTPATH

for PROBLEM in $PROBLEMS; do

  evaluation="testTT"
  evaluationOld="testTTold"

  # This strips any prefix from the data folder
  # DATANAME=${PROBLEM/*_/}
  DATANAME=${DATASET}

  # Prepares the folder for the problem results
  mkdir ${RESULTPATH}/${PROBLEM}

  # for SEED in $SEEDS; do

  # Prepares the folder for this execution of the problem
  RESULTFOLDER=${RESULTPATH}/${PROBLEM}  # /${SEED}
  mkdir $RESULTFOLDER
  echo "Result output to $RESULTFOLDER"

  # FULLDATA is the base file that includes probabilities
  # \FULLDATA=${DATAPATH}/${PROBLEM}/${SEED}/${DATANAME}.arff
  # FULLDATA=${DATAPATH}/${PROBLEM}/1/${DATANAME}_full.arff
  FULLDATA=${DATAPATH}/${PROBLEM}/${DATANAME}_full.arff

  # TRAINDATA has no probabilities
  # TRAINDATA=${DATAPATH}/${PROBLEM}/${SEED}/${DATANAME}_train.arff
  # TRAINDATA=${DATAPATH}/${PROBLEM}/1/${DATANAME}_train.arff
  TRAINDATA=${DATAPATH}/${PROBLEM}/${DATANAME}.arff

  # This makes a TRAINDATA file as a copy of the FULLDATA file
  # but places each copy in a separate directory
  # makeTrainSet_COPY


  # ################################

  # Start of grid search on relevant algorithms


  # Base classifiers
  # # 1
  HT="trees.HoeffdingTree"  # nothing for this test (the purpose should be to compare it to the raw one).
  # ALGORITHM=${HT}
  # ALGOTOK="HT_vanilla"
  # ${evaluation}
  # ${evaluation}Window 
  # # # 2
  HAT="trees.HoeffdingAdaptiveTree"  # nothing for this test (the purpose should be to compare it to the raw one).
  # ALGORITHM=${HAT}
  # ALGOTOK="HAT_vanilla"
  # ${evaluation}
  # ${evaluation}Window


  # elts trigger all base classfiers if the algo is base to avoid triggering extra exps later.
  INB="bayes.NaiveBayes"
  ALGORITHM=${INB}
  ALGOTOK="iNB"
  # ${evaluation}
  # ${evaluation}Window 
  if [ $1 == "base" ]; then
    ${evaluation}
    ${evaluation}Window
  fi

  HT="trees.HoeffdingTree"  # nothing for this test (the purpose should be to compare it to the raw one).
  ALGORITHM=${HT}
  ALGOTOK="HT_vanilla"
  if [ $1 == "base" ]; then
    ${evaluation}
    ${evaluation}Window
  fi

  # # # 2
  HAT="trees.HoeffdingAdaptiveTree"  # nothing for this test (the purpose should be to compare it to the raw one).
  ALGORITHM=${HAT}
  ALGOTOK="HAT_vanilla"
  if [ $1 == "base" ]; then
    ${evaluation}
    ${evaluation}Window
  fi



  # # Main algorithm
  # NA_distanceTreshs="0.5 0.75 1 1.25 1.5 1.75 2"
  # NA_driftDets="ADWINChangeDetector" # "DDM EDDM RDDM HDDM_A_Test" 
  # NA_distanceTreshs="1.25 0.75"
  # NA_distanceTreshs="1"
  # NA_distanceTreshs="1.25"
  # NA_distanceTreshs="1.5 1.75"
  # NA_distanceTreshs="2 0.5"  

  # # NA_distanceTreshs="0.5 0.75 1 1.25 1.5 1.75 2"
  NA_driftDets="DDM EDDM RDDM HDDM_A_Test ADDM"
  NA_drift_adwin_confidences="1 2 3 4 5"  # ADDM -a 1.0E-5 (mod) & -a 1.0E-3 (fast)
  gng_lambdas="50 40 30 20 10" # 50"  # 10 15 20 25 30 35 40 45 50
  gng_maxAges="200 300 400 500 100"
  NA_groupMaxSizes="5"
  NA_feature_subset_indexes1="1,3,12,15,16,17"
  NA_feature_subset_indexes2="0,1,3,5,6,12,15,16,17"
  NA_RDDM_confidences="1.8 2.0 2.2 2.258"
  NA_baseClassifier=${INB}
  echo ${INB}
  # groupReplacementPolicy="" not now
  # groupSelectionPolicy="" not now
  mahalanobis_init_set="${mahalanobis_init_set//;/;${DATAPATH}/${PROBLEM}/}"

  if [ $1 == "NA_1" ] || [ $1 == "NA_2" ]; then  # before Aug 2020
  # for NA_distanceThresh in $NA_distanceTreshs; do
    NA_distanceThresh=$2
    gng_maxAge=$6
    NA_driftDet=$7
    for gng_lambda in $gng_lambdas; do
      # for gng_maxAge in $gng_maxAges; do
        for NA_groupMaxSize in $NA_groupMaxSizes; do
          # for NA_driftDet in $NA_driftDets; do
            if [ "${NA_driftDet}" == "ADDM" ]; then
                for NA_drift_adwin_confidence in $NA_drift_adwin_confidences; do
                  # Not saving Event files (-g) on grid search for the sake of saving disk space.
                  if [ "${NA_drift_adwin_confidence}" == "1" ]; then
                    warningDetDegree="0.5"
                    warningDet="0.3162"
                  else
                    warningDetDegree=`expr ${NA_drift_adwin_confidence} - 1`
                    warningDet="1.0E-${warningDetDegree}"
                  fi
                  if [ "${NA_driftDet}" == "ADDM" ]; then
                    driftDetection="${NA_driftDet} -a 1.0E-${NA_drift_adwin_confidence} -p ${warningDet}"
                  else
                    driftDetection=${NA_driftDet}
                  fi
                  
                  if [ $1 == "NA_1" ]; then
                    NA="meta.NA -l ${NA_baseClassifier} -D 1 -z 1 -k ${NA_feature_subset_indexes1} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${mahalanobis_init_set} -d (${driftDetection})"
                    ALGORITHM=${NA}
                    ALGOTOK="NA_iNB_${NA_groupMaxSize}_1_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${NA_driftDet}_${NA_drift_adwin_confidence}"
                    # #
                    # FILE=${ALGOTOK}.txt
                    # if [ -f "$FILE" ]; then
                    #     echo "$FILE exists."  # THEN DO NOT RUN
                    # else 
                    #     echo "$FILE does not exist."
                    #     # RUN
                    # fi
                    echo "***************"
                    echo ${DATASET}
                    echo ${mahalanobis_init_set}
                    echo ${driftDetection}
                    echo ${NA_driftDet}
                    echo '------'
                    echo ${NA}
                    echo "***************"
                    ${evaluation}
                    ${evaluation}Window 
                  fi
                  if [ $1 == "NA_2" ]; then
                    NA="meta.NA -l ${NA_baseClassifier} -D 1 -z 1 -k ${NA_feature_subset_indexes2} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${mahalanobis_init_set} -d (${driftDetection})"
                    ALGORITHM=${NA}
                    echo "***************"
                    echo ${DATASET}
                    echo ${mahalanobis_init_set}
                    echo ${driftDetection}
                    echo ${NA_driftDet}
                    echo '------'
                    echo ${NA}
                    echo "***************"
                    ALGOTOK="NA_iNB_${NA_groupMaxSize}_2_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${NA_driftDet}_${NA_drift_adwin_confidence}"
                    ${evaluation}
                    ${evaluation}Window 
                  fi
                done
            else
                driftDetection=${NA_driftDet}
                echo ${NA_driftDet}
                if [ "${NA_driftDet}" == "RDDM" ]; then
                  for RDDM_confidence in $NA_RDDM_confidences; do
                    driftDetection="${NA_driftDet} -o ${RDDM_confidence}"
                    # Common block in HDMMA too
                    if [ $1 == "NA_1" ]; then
                      NA="meta.NA -l ${NA_baseClassifier} -D 1 -z 1 -k ${NA_feature_subset_indexes1} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${mahalanobis_init_set} -d (${driftDetection})"
                      ALGORITHM=${NA}
                      echo "***************"
                      echo ${DATASET}
                      echo ${mahalanobis_init_set}
                      echo ${driftDetection}
                      echo ${NA_driftDet}
                      echo '------'
                      echo ${NA}
                      echo "***************"
                      ALGOTOK="NA_iNB_${NA_groupMaxSize}_1_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${NA_driftDet}-${RDDM_confidence}"
                      ${evaluation}
                      ${evaluation}Window 
                    fi
                    if [ $1 == "NA_2" ]; then
                      NA="meta.NA -l ${NA_baseClassifier} -D 1 -z 1 -k ${NA_feature_subset_indexes2} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${mahalanobis_init_set} -d (${driftDetection})"
                      ALGORITHM=${NA}
                      echo "***************"
                      echo ${DATASET}
                      echo ${mahalanobis_init_set}
                      echo ${driftDetection}
                      echo ${NA_driftDet}
                      echo '------'
                      echo ${NA}
                      echo "***************"
                      ALGOTOK="NA_iNB_${NA_groupMaxSize}_2_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${NA_driftDet}-${RDDM_confidence}"
                      ${evaluation}
                      ${evaluation}Window 
                    fi
                    # end of block
                  done
                else  # HDDM_A
                  # common block with RDDM
                  if [ $1 == "NA_1" ]; then
                    NA="meta.NA -l ${NA_baseClassifier} -D 1 -z 1 -k ${NA_feature_subset_indexes1} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${mahalanobis_init_set} -d (${driftDetection})"
                    ALGORITHM=${NA}
                    echo "***************"
                    echo ${DATASET}
                    echo ${mahalanobis_init_set}
                    echo ${driftDetection}
                    echo ${NA_driftDet}
                    echo '------'                    
                    echo ${NA}
                    echo "***************"
                    ALGOTOK="NA_iNB_${NA_groupMaxSize}_1_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${NA_driftDet}"
                    ${evaluation}
                    ${evaluation}Window 
                  fi
                  if [ $1 == "NA_2" ]; then
                    NA="meta.NA -l ${NA_baseClassifier} -D 1 -z 1 -k ${NA_feature_subset_indexes2} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${mahalanobis_init_set} -d (${driftDetection})"
                    ALGORITHM=${NA}
                    echo "***************"
                    echo ${DATASET}
                    echo ${mahalanobis_init_set}
                    echo ${driftDetection}
                    echo ${NA_driftDet}
                    echo '------'
                    echo ${NA}
                    echo "***************"
                    ALGOTOK="NA_iNB_${NA_groupMaxSize}_2_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${NA_driftDet}"
                    ${evaluation}
                    ${evaluation}Window 
                  fi
                  # end of block
                fi
            fi
          done
        # done
      # done
    done
  # done
  fi

  NA_driftDets="DDM EDDM RDDM HDDM_A_Test ADWINChangeDetector"
  NA_drift_adwin_confidences="1 2 3 4 5"  # ADWINChangeDetector -a 1.0E-5 (mod) & -a 1.0E-3 (fast)
  
  if [ $1 == "NA_1_2det" ] || [ $1 == "NA_2_2det" ]; then  # before Aug 2020
  # for NA_distanceThresh in $NA_distanceTreshs; do
    NA_distanceThresh=$2
    gng_maxAge=$6
    NA_driftDet=$7
    for gng_lambda in $gng_lambdas; do
      # for gng_maxAge in $gng_maxAges; do
        for NA_groupMaxSize in $NA_groupMaxSizes; do
          # for NA_driftDet in $NA_driftDets; do
            if [ "${NA_driftDet}" == "ADWINChangeDetector" ] || [ "${NA_driftDet}" == "DDM" ] || [ "${NA_driftDet}" == "EDDM" ]; then
                for NA_drift_adwin_confidence in $NA_drift_adwin_confidences; do
                  # Not saving Event files (-g) on grid search for the sake of saving disk space.
                  if [ "${NA_driftDet}" == "ADWINChangeDetector" ]; then
                    driftDetection="${NA_driftDet} -a 1.0E-${NA_drift_adwin_confidence}"
                  else
                    driftDetection=${NA_driftDet}
                  fi

                  if [ "${NA_drift_adwin_confidence}" == "1" ]; then
                    warningDetDegree="0.5"
                    warningDet="ADWINChangeDetector -a 0.3162"
                  else
                    warningDetDegree=`expr ${NA_drift_adwin_confidence} - 1`
                    warningDet="ADWINChangeDetector -a 1.0E-${warningDetDegree}"
                  fi
                  
                  if [ $1 == "NA_1_2det" ]; then
                    NA="meta.NATwoDetectors -l ${NA_baseClassifier} -d 1 -z 1 -k ${NA_feature_subset_indexes1} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${mahalanobis_init_set} -x (${driftDetection}) -p (${warningDet})"
                    ALGORITHM=${NA}
                    ALGOTOK="NAold_iNB_${NA_groupMaxSize}_1_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${NA_driftDet}_${NA_drift_adwin_confidence}"
                    echo "***************"
                    echo ${NA}
                    echo "***************"
                    ${evaluation}
                    ${evaluation}Window 
                  fi
                  if [ $1 == "NA_2_2det" ]; then
                    NA="meta.NATwoDetectors -l ${NA_baseClassifier} -d 1 -z 1 -k ${NA_feature_subset_indexes2} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${mahalanobis_init_set} -x (${driftDetection}) -p (${warningDet})"
                    ALGORITHM=${NA}
                    echo "***************"
                    echo ${NA}
                    echo "***************"
                    ALGOTOK="NAold_iNB_${NA_groupMaxSize}_2_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${NA_driftDet}_${NA_drift_adwin_confidence}"
                    ${evaluation}
                    ${evaluation}Window 
                  fi
                done
            else
                driftDetection=${NA_driftDet}
                echo ${NA_driftDet}
                if [ "${NA_driftDet}" == "RDDM" ]; then
                  warningDet="RDDM -o 1.773 -w 1.455 -n 70 -x 30000"   # change detection at default warning interval (1.773)
                  for RDDM_confidence in $NA_RDDM_confidences; do
                    driftDetection="${NA_driftDet} -o ${RDDM_confidence}"
                    # Common block in HDMMA too
                    if [ $1 == "NA_1_2det" ]; then
                      NA="meta.NATwoDetectors -l ${NA_baseClassifier} -d 1 -z 1 -k ${NA_feature_subset_indexes1} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${mahalanobis_init_set} -x (${driftDetection}) -p (${warningDet})"
                      ALGORITHM=${NA}
                      echo "***************"
                      echo ${NA}
                      echo "***************"
                      ALGOTOK="NAold_iNB_${NA_groupMaxSize}_1_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${NA_driftDet}-${RDDM_confidence}"
                      ${evaluation}
                      ${evaluation}Window 
                    fi
                    if [ $1 == "NA_2_2det" ]; then
                      NA="meta.NA -l ${NA_baseClassifier} -d 1 -z 1 -k ${NA_feature_subset_indexes2} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${mahalanobis_init_set} -x (${driftDetection}) -p (${warningDet})"
                      ALGORITHM=${NA}
                      echo "***************"
                      echo ${NA}
                      echo "***************"
                      ALGOTOK="NA_iNB_${NA_groupMaxSize}_2_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${NA_driftDet}-${RDDM_confidence}"
                      ${evaluation}
                      ${evaluation}Window 
                    fi
                    # end of block
                  done
                else  # HDDM_A
                  warningDet="HDDM_A_Test -d 0.005 -w 0.01"  # default warning interval on change detectors, now for drift
                  echo ${warningDet}
                  # warningDet="ADWINChangeDetector -a 1.0E-3"  # default
                  # common block with RDDM
                  if [ $1 == "NA_1_2det" ]; then
                    NA="meta.NATwoDetectors -l ${NA_baseClassifier} -d 1 -z 1 -k ${NA_feature_subset_indexes1} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${mahalanobis_init_set} -x (${driftDetection}) -p (${warningDet})"
                    ALGORITHM=${NA}
                    echo "***************"
                    echo ${NA}
                    echo "***************"
                    ALGOTOK="NAold_iNB_${NA_groupMaxSize}_1_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${NA_driftDet}"
                    ${evaluation}
                    ${evaluation}Window 
                  fi
                  if [ $1 == "NA_2_2det" ]; then
                    NA="meta.NATwoDetectors -l ${NA_baseClassifier} -d 1 -z 1 -k ${NA_feature_subset_indexes2} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${mahalanobis_init_set} -x (${driftDetection}) -p (${warningDet})"
                    ALGORITHM=${NA}
                    echo "***************"
                    echo ${NA}
                    echo "***************"
                    ALGOTOK="NAold_iNB_${NA_groupMaxSize}_2_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${NA_driftDet}"
                    ${evaluation}
                    ${evaluation}Window 
                  fi
                  # end of block
                fi
            fi
          done
        # done
      # done
    done
  # done
  fi

  # ##############################
  # Competitors from the SoTA
  # ##############################

  # # CPF / ECPF
  CPF_baseClassifier=${INB}
  # CPF_drift_adwin_confidences="5 3"  # ADWINChangeDetector -a 1.0E-5 (mod) & -a 1.0E-3 (fast)
  CPF_drift_adwin_confidences="5 3 2"  # ADWINChangeDetector -a 1.0E-5 (mod) & -a 1.0E-3 (fast)
  CPF_driftDets="EDDM ADWINChangeDetector RDDM HDDM_A_Test"  # DDM
  CPF_fadeModelss=("-f" "")
  # CPF_ms="0.85 0.925 0.95 0.975 0.99 0.9"  
  CPF_min_buffer_sizes="30 60 120 180 240"
  CPF_f="15" # comparable to max group size x number of groups
  if [ $1 == "CPF" ]; then
    CPF_m=$2
    # CPF_driftDet=$6
    for CPF_fadeModels in "${CPF_fadeModelss[@]}"; do
      # for CPF_m in $CPF_ms; do
        for CPF_driftDet in $CPF_driftDets; do
          if [ "${CPF_driftDet}" == "ADWINChangeDetector" ]; then
            for CPF_drift_adwin_confidence in $CPF_drift_adwin_confidences; do
              driftDetection="${CPF_driftDet} -a 1.0E-${CPF_drift_adwin_confidence}"
              for CPF_min_buffer_size in $CPF_min_buffer_sizes; do
                CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
                ALGORITHM=${CPF}
                ALGOTOK="CPF_iNB_${CPF_fadeModels}_${CPF_f}_${CPF_m}_${CPF_min_buffer_size}_${CPF_driftDet}_${CPF_drift_adwin_confidence}"
                ${evaluation}
                ${evaluation}Window 
              done
              ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
              ALGORITHM=${ECPF}
              ALGOTOK="ECPF_iNB_${CPF_fadeModels}_${CPF_f}_${CPF_m}_${CPF_driftDet}_${CPF_drift_adwin_confidence}"
              ${evaluation}
              ${evaluation}Window 
            done
          else
              driftDetection=${CPF_driftDet}
              for CPF_min_buffer_size in $CPF_min_buffer_sizes; do
                CPF="meta.CPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -b ${CPF_min_buffer_size}"
                ALGORITHM=${CPF}
                ALGOTOK="CPF_iNB_${CPF_fadeModels}_${CPF_f}_${CPF_m}_${CPF_min_buffer_size}_${CPF_driftDet}"
                ${evaluation}
                ${evaluation}Window 
              done
              ECPF="meta.ECPF -l ${CPF_baseClassifier} ${CPF_fadeModels} -p ${CPF_f} -d (${driftDetection}) -m ${CPF_m} -c 1" 
              ALGORITHM=${ECPF}
              ALGOTOK="ECPF_iNB_${CPF_fadeModels}_${CPF_f}_${CPF_m}_${CPF_driftDet}"
              ${evaluation}
              ${evaluation}Window 
          fi
        done
      # done
    done
  fi


  # # # # RCD
  RCD_baseClassifier=${INB}
  # RCD_driftDets="DDM EDDM ADWINChangeDetector RDDM HDDM_A_Test"
  RCD_drift_adwin_confidences="5 3 2"  # ADWINChangeDetector -a 1.0E-5 (mod) & -a 1.0E-3 (fast)
  RCD_ks="1 3 5 7" #  the number of nearest neighbors (only odd values tested to avoid ties)
  RCD_cs="1 15" # number of classifiers
  RCD_ss="0.01 0.05" #  the significance value (p-value)
  # RCD_bs="100 200 300 400 500"  # the maximum amount of instances to represent each distribution. 
  RCD_ts="100 200 300 400 500" # constraint >= b  ( the rate the tests will be made in the testing phase)
  if [ $1 == "RCD" ]; then
  # for RCD_b in $RCD_bs; do
    RCD_b=$2
    RCD_driftDet=$6 
    for RCD_t in $RCD_ts; do
      # RCD_t >= RCD_b
      if [ $RCD_t -ge $RCD_b ]; then
        for RCD_s in $RCD_ss; do
          for RCD_k in $RCD_ks; do
            for RCD_c in $RCD_cs; do
              # for RCD_driftDet in $RCD_driftDets; do
                if [ "${RCD_driftDet}" == "ADWINChangeDetector" ]; then
                    for RCD_drift_adwin_confidence in $RCD_drift_adwin_confidences; do
                      driftDetection="${RCD_driftDet} -a 1.0E-${RCD_drift_adwin_confidence}"
                      RCD="meta.RCD -c ${RCD_c} -l ${RCD_baseClassifier} -a (KNN -k ${RCD_k}) -s ${RCD_s} -b ${RCD_b} -t ${RCD_t} -d (${driftDetection})" # (1 tree)
                      ALGORITHM=${RCD}
                      echo "***************"
                      echo ${RCD}
                      echo "***************"
                      ALGOTOK="RCD_iNB_${RCD_c}_${RCD_k}_${RCD_s}_${RCD_b}_${RCD_t}_${RCD_driftDet}_${RCD_drift_adwin_confidence}"
                      ${evaluation}
                      ${evaluation}Window 
                    done
                else
                    driftDetection=${RCD_driftDet}
                    RCD="meta.RCD -c ${RCD_c} -l ${RCD_baseClassifier} -a (KNN -k ${RCD_k}) -s ${RCD_s} -b ${RCD_b} -t ${RCD_t} -d ${driftDetection}" # (1 tree)
                    ALGORITHM=${RCD}
                    ALGOTOK="RCD_iNB_${RCD_c}_${RCD_k}_${RCD_s}_${RCD_b}_${RCD_t}_${RCD_driftDet}"
                    ${evaluation}
                    ${evaluation}Window 
                fi
              # done
            done
          done
        done
      fi
    done
  # done
  fi

  # TODO
  # PEARL
  # lossy_counting="True False"
  # # tree_transition="they dont say what do they use/another paper maybe?"
  # ##########
  # # TODO MOVE
  # if [ $1 == "PEARL" ]; then
  # PEARL="TODO - In Python"
  # fi
  # # TODO RUN


  # ########################
  # Ensembles with only one active classifier
  # #########################

  # ARF
  ARF_driftDets="DDM EDDM ADWINChangeDetector RDDM HDDM_A_Test" # TODO: format properly
  ARF_drift_adwin_confidences="5 3 2"  # ADWINChangeDetector -a 1.0E-5 (mod) & -a 1.0E-3 (fast)
  if [ $1 == "ARF" ]; then
  for ARF_driftDet in $ARF_driftDets; do
    if [ "${ARF_driftDet}" == "ADWINChangeDetector" ]; then
        for ARF_drift_adwin_confidence in $ARF_drift_adwin_confidences; do
          driftDetection="${ARF_driftDet} -a 1.0E-${ARF_drift_adwin_confidence}"
          warningDetDegree=`expr ${ARF_drift_adwin_confidence} - 1`
          warningDet="ADWINChangeDetector -a 1.0E-${warningDetDegree}"
          ARF="meta.AdaptiveRandomForestNoBagging -l (ARFHoeffdingTree -k 5 -e 2000000 -g 50 -c 0.01) -x (${driftDetection}) -p (${warningDet})" # (1 tree - no bagging)
          echo "***************"
          echo ${ARF}
          echo "***************"
          ALGORITHM=${ARF}
          ALGOTOK="ARF_AHT_1_${ARF_driftDet}_${ARF_drift_adwin_confidence}"
          ${evaluation}
          ${evaluation}Window 
        done
    else
        driftDetection=${ARF_driftDet}
        warningDet="ADWINChangeDetector -a 1.0E-3"  # default
        ARF="meta.AdaptiveRandomForestNoBagging -l (ARFHoeffdingTree -k 5 -e 2000000 -g 50 -c 0.01) -x (${driftDetection}) -p (${warningDet})" # (1 tree - no bagging)
        ALGORITHM=${ARF}
        echo "***************"
        echo ${ARF}
        echo "***************"
        ALGOTOK="ARF_AHT_1_${ARF_driftDet}"
        ${evaluation}
        ${evaluation}Window 
    fi
  done
  fi

  # # Accuracy Uptaded Ensemble
  AUE_baseClassifier=${INB}
  # AUE_chunksizes="100 200 300 400 500 1000"  # batch one. 1000 was the best.
  # AUE_chunksizes="1000 2000 3000 4000 5000 6000 7000 8000 9000 10000"  # batch two. 2000-3000 were the best
  AUE_chunksizes="200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000"  # batch one. 1000 was the best.
  AUE_n_memberss="1"
  if [ $1 == "AUE" ]; then
  for AUE_chunksize in $AUE_chunksizes; do
    for AUE_n_members in $AUE_n_memberss; do
      AUE="meta.AccuracyUpdatedEnsemble -n ${AUE_n_members} -l ${AUE_baseClassifier} -c ${AUE_chunksize}" # (1 tree)
      ALGORITHM=${AUE}
      ALGOTOK="AUE_iNB_${AUE_n_members}_${AUE_chunksize}"
      ${evaluation}
      ${evaluation}Window 
    done
  done
  fi

  # # DWM
  DWM_baseClassifier=${INB}
  # DWM_periods="1 25 50 75 100"
  DWM_betas="0.25 0.5 0.75"
  # DWM_thetas="0.005 0.01 0.025 0.05"
  DWM_maxExpertss="2 5 15 9999999"
  if [ $1 == "DWM" ]; then
    # for DWM_period in $DWM_periods; do
      DWM_period=$2
      DWM_theta=$6
      for DWM_beta in $DWM_betas; do
        # for DWM_theta in $DWM_thetas; do
          for DWM_maxExperts in $DWM_maxExpertss; do
            DWM="meta.DynamicWeightedMajority -l ${DWM_baseClassifier} -p ${DWM_period} -b ${DWM_beta} -t ${DWM_theta} -e ${DWM_maxExperts}" # (1 tree)
            ALGORITHM=${DWM}
            ALGOTOK="DWM_iNB_${DWM_maxExperts}_${DWM_period}_${DWM_beta}_${DWM_theta}"
            ${evaluation}
            ${evaluation}Window 
          done
        # done
      done
    # done
  fi

  # # Learn.NSE++
  LNSE_baseClassifier=${INB}
  # # LNSE_sigmoids="0.1 0.25 0.5 0.75" # this is at the paper but not in the MOA version
  # LNSE_sigSlopes="5 10 15 20"
  LNSE_sigCrossPoints="5 10 15 20"
  # LNSE_periods="10 20 40 60 80 100"
  LNSE_ensemble_sizes="1"
  # if [ $1 == "LNSE" ]; then
  # ### for LNSE_sigmoid in $LNSE_sigmoids; do
  #  LNSE_period=$2
  #  LNSE_sigSlope=$6
  #   # for LNSE_sigSlope in $LNSE_sigSlopes; do
  #     for LNSE_sigCrossPoint in $LNSE_sigCrossPoints; do
  #       # LNSE_sigCrossPoint >= LNSE_sigSlope
  #       if [ $LNSE_sigCrossPoint -ge $LNSE_sigSlope ]; then
  #         # for LNSE_period in $LNSE_periods; do
  #           for LNSE_ensemble_size in $LNSE_ensemble_sizes; do
  #             LearnNSE="meta.LearnNSE -l ${LNSE_baseClassifier} -e ${LNSE_ensemble_size} -p ${LNSE_period} -a ${LNSE_sigSlope} -b ${LNSE_sigCrossPoint}"  # (1 tree)
  #             ALGORITHM=${LearnNSE}
  #             ALGOTOK="LNSE_iNB_${LNSE_ensemble_size}_${LNSE_period}_${LNSE_sigSlope}_${LNSE_sigCrossPoint}"
  #             ${evaluation}
  #             ${evaluation}Window 
  #           done
  #         # done
  #       fi
  #     done
  #   # done
  # fi


  # TBD
  # if [ $1 == "EIS" ]; then
  # # EIS="Intelligent Systems?
  # fi

done
