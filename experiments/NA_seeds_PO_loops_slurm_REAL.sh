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
# DATAPATH=${DATA_PATH}"/synthetic" 


# Datasets related
PROBLEMS=$3
DATASET=$4
mahalanobis_init_set=$5
MAX_INSTANCES=1000000

SEED=${15}
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

  ## Only export EVENTS file in results for drift analysis. 
  # Then zip it and remove original.
  EVENTS_FILE=$RESULTFOLDER'/'$PROBLEM'_'$ALGOTOK'_EVENTS'
  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_dump.txt
  # java -cp $MOALOC/moa.jar:$WEKALOC/weka.jar:$EXTRAPATH -javaagent:$JAVAAGENT moa.DoTask \
  # echo "%%%%%%%%%%%%%%%%%%%%"
  # echo "java -cp $EXTRAPATH:$MOA_DEV -javaagent:$JAVAAGENT moa.DoTask \
  # EvaluateInterleavedTestThenTrain \
  # -l (${ALGORITHM}) \
  # -s (ArffFileStream -f ${TRAINDATA} ) \
  # -d ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_dump.txt \
  # -r ${SEED} \
  # -f 10 "
  # echo "%%%%%%%%%%%%%%%%%%%%"

  java -cp $EXTRAPATH:$MOA_DEV -javaagent:$JAVAAGENT moa.DoTask \
  "EvaluateInterleavedTestThenTrain \
  -l (${ALGORITHM} -e ${EVENTS_FILE}.txt) \
  -s (ArffFileStream -f ${TRAINDATA} ) \
  -i ${MAX_INSTANCES} \
  -f 10 " \
  > ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt
    # -r ${SEED} \
  # -O ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_task.txt \
  # zip "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.zip" "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt"
  # rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt
  # mv weka.log ${RESULTPATH}/${DATANAME}_${ALGOTOK}.log
  zip $EVENTS_FILE'.zip' $EVENTS_FILE'.txt'
  rm $EVENTS_FILE'.txt'
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

  ## Only export EVENTS file in results for drift analysis. 
  # Then zip it and remove original.
  EVENTS_FILE=$RESULTFOLDER'/'$PROBLEM'_'$ALGOTOK'_EVENTS_w'${WSIZE}
  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}_dump.txt
  java -cp $EXTRAPATH:$MOA_DEV -javaagent:$JAVAAGENT moa.DoTask \
  "EvaluateInterleavedTestThenTrain \
    -l (${ALGORITHM} -e ${EVENTS_FILE}.txt) \
    -s (ArffFileStream -f ${TRAINDATA}) \
    -i ${MAX_INSTANCES} \
    -e (WindowClassificationPerformanceEvaluator -w ${WSIZE}) \
    -f 10 " \
  > ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.txt
      # -r ${SEED} \
  # -O ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}_task.txt \
  #  zip "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.zip" "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.txt"
  #  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.txt
  zip $EVENTS_FILE'.zip' $EVENTS_FILE'.txt'
  rm $EVENTS_FILE'.txt'
}

testTTC () {
  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt
  # java -cp $MOALOC/moa.jar:$WEKALOC/weka.jar:$EXTRAPATH -javaagent:$JAVAAGENT moa.DoTask \
  java -cp $EXTRAPATH:$MOA_DEV -javaagent:$JAVAAGENT moa.DoTask \
  "EvaluateInterleavedChunks \
    -l (${ALGORITHM} -g) \
    -s (ArffFileStream -f ${TRAINDATA}) \
    -i ${MAX_INSTANCES} \
    -d ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_dump.txt \
    -c ${block_train} \
    -f ${block_test}" \
  > ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt
      # -r ${SEED} \

  #   -O ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_task.txt \
  #  zip "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.zip" "${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt"
  #  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}.txt
}

testTTCWindow () {
  rm ${RESULTFOLDER}/${DATANAME}_${ALGOTOK}_w${WSIZE}.txt
  # java -cp $MOALOC/moa.jar:$WEKALOC/weka.jar:$EXTRAPATH -javaagent:$JAVAAGENT moa.DoTask \
  java -cp $EXTRAPATH:$MOA_DEV -javaagent:$JAVAAGENT moa.DoTask \
  "EvaluateInterleavedChunks \
    -l (${ALGORITHM} -g) \
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
    -l (${ALGORITHM} -g) \
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
    -l (${ALGORITHM} -g) \
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
  RESULTFOLDER=${RESULTPATH}/${PROBLEM}/${SEED}
  mkdir $RESULTFOLDER
  echo "Result output to $RESULTFOLDER"

  # FULLDATA is the base file that includes probabilities
  # \FULLDATA=${DATAPATH}/${PROBLEM}/${SEED}/${DATANAME}.arff
  # FULLDATA=${DATAPATH}/${PROBLEM}/1/${DATANAME}_full.arff
  FULLDATA=${DATAPATH}/${PROBLEM}/${DATANAME}_full.arff

  # TRAINDATA has no probabilities
  TRAINDATA=${DATAPATH}/${PROBLEM}/${SEED}/${DATANAME}.arff
  # TRAINDATA=${DATAPATH}/${PROBLEM}/1/${DATANAME}_train.arff
  # TRAINDATA=${DATAPATH}/${PROBLEM}/${DATANAME}.arff

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
    # ${evaluation}
    ${evaluation}Window
  fi

  HT="trees.HoeffdingTree"  # nothing for this test (the purpose should be to compare it to the raw one).
  ALGORITHM=${HT}
  ALGOTOK="HT_vanilla"
  if [ $1 == "base" ]; then
    # ${evaluation}
    ${evaluation}Window
  fi

  # # # 2
  HAT="trees.HoeffdingAdaptiveTree"  # nothing for this test (the purpose should be to compare it to the raw one).
  ALGORITHM=${HAT}
  ALGOTOK="HAT_vanilla"
  if [ $1 == "base" ]; then
    # ${evaluation}
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
  # NA_driftDets="EDDM RDDM ADDM HDDM_A" # DDM EDDM RDDM HDDM_A_Test ADDM"
  # NA_drift_adwin_confidences="1 2 3 4 5"  # ADDM -a 1.0E-5 (mod) & -a 1.0E-3 (fast)
  gng_lambdas="5" #"50 40 30 20 10" # 50"  # 10 15 20 25 30 35 40 45 50
  gng_maxAges="200 300" #400 500 100"
  NA_groupMaxSizes="5"
  NA_feature_subset_indexes1="1,3,12,15,16,17"
  NA_feature_subset_indexes2="0,1,3,5,6,12,15,16,17"
  NA_RDDM_confidences="1.8 2.0 2.2 2.258"
  # groupReplacementPolicy="" not now
  # groupSelectionPolicy="" not now
  mahalanobis_init_set="${mahalanobis_init_set//;/;${DATAPATH}/${PROBLEM}/${SEED}/}"

  if [ $1 == "NA_1" ] || [ $1 == "NA_2" ]; then  # before Aug 2020
  # for NA_distanceThresh in $NA_distanceTreshs; do
    NA_distanceThresh=$2
    gng_maxAge=$6
    NA_driftDet=$(echo $7 | sed "s/'/ /g")
    trainOnWarn=$(echo $8 | sed "s/'/ /g")
    trainEND=$(echo $9 | sed "s/'/ /g")
    preTrain=$(echo ${10} | sed "s/[N]//g")
    wwMinThresh=${11} 
    minInstTrainedForInsertion=${12} 
    insExamplesOnly=$(echo ${13} | sed "s/'/ /g")
    driftWarnDiffs="0.670 0.970" # based on isolated tests
    baseClassifier=${14}
    if [ $baseClassifier == "NB" ]; then
      NA_baseClassifier=${INB}
      NA_baseClassifierName="iNB"
    fi
    if [ $baseClassifier == "HT" ]; then
      NA_baseClassifier=${HT}
      NA_baseClassifierName="HT"
    fi

    # Hardcoded ones by now...
    forceEarlyDrifts="-F"  # always on by now 
    alwaysTransitToBlankClassifAtEarlyDrift="" # always off by now 
    instanceThreshForInsertions=""  # always off by now  # "N -7500"  # TO BE ADDED TO ALGO WITH -N

    echo "START PARAMETERS"
    echo "========="
    echo "DATASET "${DATASET}
    echo "NA_distanceThresh "$NA_distanceThresh
    echo "gng_maxAge "$gng_maxAge
    echo "NA_driftDet "$NA_driftDet
    echo "trainOnWarn "$trainOnWarn
    echo "trainEND "$trainEND
    echo "preTrain "$preTrain
    echo "wwMinThresh "$wwMinThresh
    echo "minInstTrainedForInsertion "$minInstTrainedForInsertion
    echo "inserExamplesOnly "$insExamplesOnly
    echo "baseClassifier "$baseClassifier
    echo '------'
    echo "END PARAMETERS"
    for gng_lambda in $gng_lambdas; do
      # for gng_maxAge in $gng_maxAges; do
        for NA_groupMaxSize in $NA_groupMaxSizes; do
          # for NA_driftDet in $NA_driftDets; do
            if [ "${NA_driftDet}" == "ADDM" ]; then
            # IMP: APPLY LAST PARAMS HERE FOR ADDM TOO (ONLY DONE FOR EDDM TILL NOW...)
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
                  driftDetectionAlgo=$(echo ${driftDetection} | sed 's/ //g')
                  
                  if [ $1 == "NA_1" ]; then
                    NA="meta.NA -l (${NA_baseClassifier}) ${forceEarlyDrifts} ${alwaysTransitToBlankClassifAtEarlyDrift} ${instanceThreshForInsertions} ${trainOnWarn} ${trainEND} ${insExamplesOnly} ${preTrain} -w ${wwMinThresh} -I ${minInstTrainedForInsertion} -D 1 -z 1 -k ${NA_feature_subset_indexes1} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${SEED}/${mahalanobis_init_set} -d (${driftDetection})"
                    ALGORITHM=${NA}                                                                                                                        
                    ALGOTOK=$(echo "NA_${NA_baseClassifierName}_${NA_groupMaxSize}_1_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${driftDetectionAlgo}-${NA_drift_adwin_confidence}_${trainOnWarn}_${trainEND}_${preTrain}_${wwMinThresh}_${minInstTrainedForInsertion}_${forceEarlyDrifts}_${alwaysTransitToBlankClassifAtEarlyDrift}_${instanceThreshForInsertions}_${insExamplesOnly}" | sed 's/ //g')
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
                    echo "]]]]]]]"
                    echo '------'
                    echo ${NA}
                    echo "***************"
                    ${evaluation}
                    ${evaluation}Window 
                  fi
                  if [ $1 == "NA_2" ]; then
                    NA="meta.NA -l (${NA_baseClassifier}) ${forceEarlyDrifts} ${alwaysTransitToBlankClassifAtEarlyDrift} ${instanceThreshForInsertions} ${trainOnWarn} ${insExamplesOnly} ${trainEND} ${preTrain} -w ${wwMinThresh} -I ${minInstTrainedForInsertion} -D 1 -z 1 -k ${NA_feature_subset_indexes2} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${SEED}/${mahalanobis_init_set} -d (${driftDetection})"
                    ALGORITHM=${NA}
                    echo "***************"
                    echo ${DATASET}
                    echo ${mahalanobis_init_set}
                    echo ${driftDetection}
                    echo ${NA_driftDet}
                    echo '------'
                    echo ${NA}
                    echo "***************"
                    ALGOTOK=$(echo "NA_${NA_baseClassifierName}_${NA_groupMaxSize}_2_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${driftDetectionAlgo}-${NA_drift_adwin_confidence}_${trainOnWarn}_${trainEND}_${preTrain}_${wwMinThresh}_${minInstTrainedForInsertion}_${forceEarlyDrifts}_${alwaysTransitToBlankClassifAtEarlyDrift}_${instanceThreshForInsertions}_${insExamplesOnly}" | sed 's/ //g')
                    ${evaluation}
                    ${evaluation}Window 
                  fi
                done
            # END if ADDM
            else
                driftDetection=${NA_driftDet}
                echo ${NA_driftDet}
                if [ "${NA_driftDet}" == "RDDM" ]; then
                  for driftWarnDiff in $driftWarnDiffs; do
                    for RDDM_confidence in $NA_RDDM_confidences; do
                      warningDet=$(awk -v drift=${RDDM_confidence} -v warndiff=${driftWarnDiff} 'BEGIN{print drift - warndiff}')
                      # warningDet=`expr ${RDDM_confidence} - ${driftWarnDiff}`
                      echo $warningDet
                      driftDetection="${NA_driftDet} -o ${RDDM_confidence} -w ${warningDet}"
                      echo $driftDetection 
                      echo "LALALA"
                      driftDetectionAlgo=$(echo ${driftDetection} | sed 's/ //g') 
                      echo $RDDM_confidence
                      # Common block with RDDMA (in HDDMA too)
                      if [ $1 == "NA_1" ]; then
                        echo $driftDetection
                        
                        NA="meta.NA -l (${NA_baseClassifier}) ${forceEarlyDrifts} ${alwaysTransitToBlankClassifAtEarlyDrift} ${instanceThreshForInsertions} ${insExamplesOnly} ${trainOnWarn} ${trainEND} ${preTrain} -w ${wwMinThresh} -I ${minInstTrainedForInsertion} -D 1 -z 1 -k ${NA_feature_subset_indexes1} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${SEED}/${mahalanobis_init_set} -d (${driftDetection})"
                        ALGORITHM=${NA}
                        echo "***************"
                        # echo ${DATASET}
                        # echo ${mahalanobis_init_set}
                        # echo ${driftDetection}
                        # echo ${NA_driftDet}
                        # echo "[[[[[[[[[[["
                        # echo '------'
                        echo ${NA}
                        echo "***************"
                        ALGOTOK=$(echo "NA_${NA_baseClassifierName}_${NA_groupMaxSize}_1_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${driftDetectionAlgo}-${RDDM_confidence}_${trainOnWarn}_${trainEND}_${preTrain}_${wwMinThresh}_${minInstTrainedForInsertion}_${forceEarlyDrifts}_${alwaysTransitToBlankClassifAtEarlyDrift}_${instanceThreshForInsertions}_${insExamplesOnly}" | sed 's/ //g')
                        ${evaluation}
                        ${evaluation}Window 
                      fi
                      if [ $1 == "NA_2" ]; then
                        NA="meta.NA -l (${NA_baseClassifier}) ${forceEarlyDrifts} ${alwaysTransitToBlankClassifAtEarlyDrift} ${instanceThreshForInsertions} ${insExamplesOnly} ${trainOnWarn} ${trainEND} ${preTrain} -w ${wwMinThresh} -I ${minInstTrainedForInsertion} -D 1 -z 1 -k ${NA_feature_subset_indexes2} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${SEED}/${mahalanobis_init_set} -d (${driftDetection})"
                        ALGORITHM=${NA}
                        echo "***************"
                        echo ${DATASET}
                        echo ${mahalanobis_init_set}
                        echo ${driftDetection}
                        echo ${NA_driftDet}
                        echo '------'
                        echo ${NA}
                        echo "***************"
                        ALGOTOK=$(echo "NA_${NA_baseClassifierName}_${NA_groupMaxSize}_2_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${driftDetectionAlgo}-${RDDM_confidence}_${trainOnWarn}_${trainEND}_${preTrain}_${wwMinThresh}_${minInstTrainedForInsertion}_${forceEarlyDrifts}_${alwaysTransitToBlankClassifAtEarlyDrift}_${instanceThreshForInsertions}_${insExamplesOnly}" | sed 's/ //g')
                        ${evaluation}
                        ${evaluation}Window 
                      fi
                      # end of block
                    done
                  done
                else  # HDDM_A
                  # common block with HDDMA
                  if [ $1 == "NA_1" ]; then 
                    driftDetectionAlgo=$(echo ${driftDetection} | sed 's/ //g')
                    NA="meta.NA -l (${NA_baseClassifier}) ${forceEarlyDrifts} ${alwaysTransitToBlankClassifAtEarlyDrift} ${instanceThreshForInsertions} ${trainOnWarn} ${insExamplesOnly} ${trainEND} ${preTrain} -w ${wwMinThresh} -I ${minInstTrainedForInsertion} -D 1 -z 1 -k ${NA_feature_subset_indexes1} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${SEED}/${mahalanobis_init_set} -d (${driftDetection})"
                    ALGORITHM=${NA}
                    echo "***************"
                    echo ${DATASET}
                    echo ${mahalanobis_init_set}
                    echo ${driftDetection}
                    echo ${NA_driftDet}
                    echo "zzzzzzz"
                    echo '------'
                    echo ${NA}
                    echo "***************"
                    ALGOTOK=$(echo "NA_${NA_baseClassifierName}_${NA_groupMaxSize}_1_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${driftDetectionAlgo}_${trainOnWarn}_${trainEND}_${preTrain}_${wwMinThresh}_${minInstTrainedForInsertion}_${forceEarlyDrifts}_${alwaysTransitToBlankClassifAtEarlyDrift}_${instanceThreshForInsertions}_${insExamplesOnly}" | sed 's/ //g')
                    ${evaluation}
                    ${evaluation}Window 
                  fi
                  if [ $1 == "NA_2" ]; then
                    driftDetectionAlgo=$(echo ${driftDetection} | sed 's/ //g')
                    NA="meta.NA -l (${NA_baseClassifier}) ${forceEarlyDrifts} ${alwaysTransitToBlankClassifAtEarlyDrift} ${instanceThreshForInsertions} ${trainOnWarn} ${insExamplesOnly} ${trainEND} ${preTrain} -w ${wwMinThresh} -I ${minInstTrainedForInsertion} -D 1 -z 1 -k ${NA_feature_subset_indexes2} -M ${NA_groupMaxSize} -t ${NA_distanceThresh} -c (GNG -l ${gng_lambda} -m ${gng_maxAge} -c 2147483647 -b) -o ${DATAPATH}/${PROBLEM}/${SEED}/${mahalanobis_init_set} -d (${driftDetection})"
                    ALGORITHM=${NA}
                    echo "***************"
                    echo ${DATASET}
                    echo ${mahalanobis_init_set}
                    echo ${driftDetection}
                    echo ${NA_driftDet}
                    echo '------'
                    echo ${NA}
                    echo "***************"
                    ALGOTOK=$(echo "NA_${NA_baseClassifierName}_${NA_groupMaxSize}_2_${NA_distanceThresh}_${gng_lambda}_${gng_maxAge}_${driftDetectionAlgo}_${trainOnWarn}_${trainEND}_${preTrain}_${wwMinThresh}_${minInstTrainedForInsertion}_${forceEarlyDrifts}_${alwaysTransitToBlankClassifAtEarlyDrift}_${instanceThreshForInsertions}_${insExamplesOnly}" | sed 's/ //g')
                    ${evaluation}
                    ${evaluation}Window 
                  fi
                  # end of block
                fi
            fi
          # done
        done
      # done
    done
  fi

done

