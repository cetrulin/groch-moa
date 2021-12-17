/*
 *    GroCH.java
 *
 *    @author Andres Leon Suarez Cetrulo (suarezcetrulo at gmail dot com)
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */
package moa.classifiers.meta;

//import moa.classifiers.meta.MahalanobisDistanceSingleMatrix;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.github.javacliparser.StringOption;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;

import moa.core.DoubleVector;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.Utils;
import moa.options.ClassOption;
import weka.core.MahalanobisDistance;

// import weka.gui.beans.Clusterer;
import weka.core.converters.ArffLoader.ArffReader;
import moa.classifiers.Classifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.drift.DriftDetectionMethodClassifierExt;
import moa.classifiers.igngsvm.gng.GNG;
import moa.classifiers.igngsvm.gng.GUnit;
import moa.classifiers.lazy.neighboursearch.EuclideanDistanceModified;
import moa.classifiers.lazy.neighboursearch.LinearNNSearchModified;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.DynamicWindowClassificationPerformanceEvaluator;
import moa.evaluation.LearningPerformanceEvaluator;
import moa.classifiers.core.driftdetection.NAChangeDetector;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstanceImpl;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;

/**
 * Evolving Pool of Classifiers with History
 *
 * @author Andres Leon Suarez Cetrulo (suarezcetrulo at gmail dot com)
 * @version $Revision: 1 $
 * IMP: GroCH deals with structured streams that do not vary their number of attributes overtime.
 */
public class GroCH extends DriftDetectionMethodClassifierExt {

	int MIN_LIMIT_PRINTS = 0;
	int MAX_LIMIT_PRINTS = 1980;

	@Override
	public String getPurposeString() {
		return "NA from Suarez-Cetrulo et al.";
	}

	private static final long serialVersionUID = 1L;

	/////////////
	// Options
	// -------
//	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.", Classifier.class,
//			"trees.HoeffdingTree"); // default params for hoeffding trees

//	public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
//			"Change detector for drifts and its parameters", ChangeDetector.class, "NAChangeDetector -x (ADWINChangeDetector -a 1.0E-5 -p (ADWINChangeDetector -a 1.0E-4))");
	
    
	public FlagOption saveClassifiersOnFalseAlarmOption = new FlagOption("saveClassifiersOnFalseAlarm", 'm',
			"Should the algorithm save the classifiers in the Concept History in the case of False Alarms?");  // 

	public FlagOption disableDriftDetectionOption = new FlagOption("disableDriftDetection", 'u',
			"Should the algorithm use drift detection? If disabled then bkg learner is also disabled.");

	public FlagOption disableRecurringDriftDetectionOption = new FlagOption("disableRecurringDriftDetection", 'r',
			"Should the algorithm save old learners to compare against in the future? If disabled then recurring concepts are not handled explicitly.");

	public IntOption defaultWindowOption = new IntOption("defaultWindow", 'D',
			"Number of rows by default in Dynamic Sliding Windows "
			+ "(Always 1 when windowResizePolicyOption == -1 [always growing during warning window]).", 
			50, 1, Integer.MAX_VALUE);

	public IntOption minWindowSizeOption = new IntOption("minWindowSize", 'z',
			"Minimum window size in Dynamic Sliding Windows "
			+ "(Always 1 when windowResizePolicyOption == -1 [always growing during warning window])", 
			50, 1, Integer.MAX_VALUE);
	
	public FlagOption disableResizingOfAllWindowsOption = new FlagOption("disableResizingOfAllWindows", 'b',
			"Should the internal evaluator windows for old learners be static (not to grow during the warning window)? "
			+ "These are not static automatically if windowResizePolicyOption == -1 [always growing during warning window].");

	public IntOption windowResizePolicyOption = new IntOption("windowResizePolicy", 'W',
			"-1 to consider all examples during the warning window (window always growing). "
			+ "-1 is default in NA; "
			+ "0 is default in RCARF. "
			+ "1 is default in NAsingle (initial GroCH always onepass).", 
			-1, -1, 3);
	
	public StringOption eventsLogFileOption = new StringOption("eventsLogFile", 'e',
			"File path to export events as warnings and drifts", "./NA_events_log.txt");

	public FlagOption disableEventsLogFileOption = new FlagOption("disableEventsLogFile", 'g',
			"Should export event logs to analyze them in the future? If disabled then events are not logged.");

	public IntOption logLevelOption = new IntOption("eventsLogFileLevel", 'h',
			"0 only logs drifts; 1 logs drifts + warnings; 2 logs every data example", 1, 0, 2);

	public ClassOption evaluatorOption = new ClassOption("baseClassifierEvaluator", 'f',
			"Classification performance evaluation method in each base classifier for voting.",
			LearningPerformanceEvaluator.class, "BasicClassificationPerformanceEvaluator");

	public IntOption warningWindowSizeThresholdOption = new IntOption("WarningWindowSizeThreshold", 'i',
			"Threshold for warning window size that disables a warning.", 500, 1, Integer.MAX_VALUE);
	
	public IntOption minInsertionThresholdOption = new IntOption("minInsertionThreshold", 'I',
			"An active classifier should have been trained with at least this number of examples to be inserted in the CH.",
			0, 0, Integer.MAX_VALUE);

	public IntOption maxGroupSizeOption = new IntOption("maxGroupSize", 'M',
			"Max number of classifiers allowed in group of the History.", 1, 1, Integer.MAX_VALUE);
	
	
    public IntOption periodBetweenDriftsOption = new IntOption("periodBetweenDrifts", 'O', "Size of the environments.", 1000000000, 1, Integer.MAX_VALUE);
	
    public IntOption insertionPeriodOption = new IntOption("insertionPeriod", 'N', "Max size of the environments in CH.", 1000000000, 1, Integer.MAX_VALUE);
    
//	public IntOption rowsToPreTrainMatrixOption = new IntOption("rowsToPreTrainMatrixOption", 'R',
//			"If Mahalanobis distances are used and no pretraining datasets are specified, the first rows of the data stream are only used to pretrain the matrix", 
//			10000, 1, 1000000); // TODO
	
	public FlagOption alwaysInsertClassifiersOption = new FlagOption("alwaysInsertClassifiers", 'A',
			"Should the classifiers be inserted in a group even if the retrieval is from the same group and the retrieved performs better?.");
		
	public MultiChoiceOption groupReplacementPolicyOption = new MultiChoiceOption("groupReplacementPolicy", 'P', 
	    		"Policy to replace a classifier from the Concept History Group when this reaches its maximum size.", 
	    		new String[]{"FIFO", "Least Used Model" //, "LRU", "Greatest Error Out"
	    				}, new String[]{"First In First Out", "Least Used Model. If equal, First In First Out." //, "Least Recently Used", "GreatestErrorOut"
	    						}, 0);
	
	public MultiChoiceOption priorityOfRecurringClassifiersOption = new MultiChoiceOption("priorityOfRecurringClassifiers", 'G', 
    		"Policy to replace a classifier from the Concept History Group when this reaches its maximum size.", 
    		new String[]{"Old", "New"}, new String[]{"Prioritise old", "Prioritise new"}, 1);
	// old works well using LUM (to avoid reusing new models that are as good as old models that we keep), and 'new' makes sense for FIFO.

	public FloatOption distThresholdOption = new FloatOption("distThreshold", 't',
			"Max distance allowed between topologies to be considered part of the same group. "
			+ "This value should change depending on the distance metric used, and the subset of attributes for concept similarity", 
			1.0, Float.MIN_VALUE, Float.MAX_VALUE);
	
    public MultiChoiceOption distanceMetricOption = new MultiChoiceOption("distanceMetric", 's', 
    		"Distance metric for concept similarity.", new String[]{"Euclidean", "Mahalanobis"}, 
    		new String[]{"Linear Euclidean distance.", "Correlation-based"}, 1);
	
    public MultiChoiceOption numberOfMatricesOption = new MultiChoiceOption("numberOfMatrices", 'v', 
    		"Should we have a matrix per Concept or a single matrix to escale distances? "
    		+ "(Only applicable when using Mahalanobis distances).", 
    		new String[]{"n Matrices", "1 matrix only"}, 
    		new String[]{"A matrix per group", "Common to all groups"}, 1);
	
	public IntOption minTopologySizeForDriftOption = new IntOption("minTopologySizeForDrift", 'a',
			"Minimum number of prototypes created before allowing a drift.", 1, 1, Integer.MAX_VALUE);
	
	// BY DEFAULT, SAME THAN minWindowSizeOption, AS BOTH SHOULD BE A MIN AMOUNT OF INSTACES THAT REPRESENTS WELL ENOUGH THE SET.
	public IntOption minWSizeForDriftOption = new IntOption("minWSizeForDrift", 'w',
			"Minimum number of instances in warning window W before allowing a drift.",
			20, 1, Integer.MAX_VALUE);   // changed from 1 to 20 on 19-01-2021
	
	public FlagOption updateGroupTopologiesOption = new FlagOption("updateGroupTopologies", 'j',
			"Should the topologies of groups be updated when inserting a new classifier into them? If disabled these won't be updated.");
	
	public StringOption topologyAttributeSubsetOption = new StringOption("topologyAttributeSubset", 'k', 
			"Subset of the attribute set applied to group concepts in the history, as a comma separated list "
			+ "(without spaces) of zero-indexed positional integers.", "-1,-1");
	
	public FlagOption disableTopologyLearnerOption = new FlagOption("disableTopologyLearner", 'q',
			"Should the learner for topology summaries be disabled?");
	
	public FlagOption trainOnlineFlagOption = new FlagOption("trainOnlineFlag", 'H', "Should the topology be trained on the go or only when a drift is detected?");
	
	// Parameters for the clustering algorithm   // TODO: make it a Clusterer.class
	public ClassOption topologyLearnerOption = new ClassOption("topologyLearner", 'c', "Clusterer to train.", GNG.class,
			"GNG -l 50 -m 200 -a 0.5 -d 0.995 -e 0.2 -n 0.006 -c "+Integer.MAX_VALUE+" -b");  // plain vanilla for GNG changing the stopping criteria and lambda (to the lowest possible value bearing in mind performance)
	
	public FlagOption multiPassTopologyTrainingOption = new FlagOption("multiPassTopologyTraining", 'y', 
			 "Should the topology training be one-pass, or to have training epochs until an stopping criteria is met? (instances only feed once)");
	
	public FlagOption multiClusterOption = new FlagOption("multiCluster", 'Y', 
			 "If selected, all groups with a topology inside the distance theshold are considered for Concept Similarity in a retrieval and not only the nearest one.");
	
	public FlagOption insertExamplesOnlyOption = new FlagOption("InsertExamplesOnly", 'Q', 
			 "If selected, groups will have a single classifier only and will send their training instances to it.");
	
	public FlagOption initializeConceptHistoryOption = new FlagOption("initializeConceptHistory", 'n',
			"The Concept History is initialized with pre-defined sets.");
	
	public StringOption arffToInitCHOption = new StringOption("arrfsToInitializeCH",'o',
			"ARFF files with identical feature set to main dataset, "
			+ "to initialize the Concept History and/or the Mahalanobis covariance matrix (if the option for a single matrix is selected). "
			+ "Paths in order (e.g. first will be group 1), separated by /';/'. "
			+ "Each file will initialize a group with a topology and a classifier trained with the totality of its data.", 
			"-1;-1");  // "/some/where/file1.arff;/some/where/file2.arff;/some/where/file3.arff;/some/where/file4.arff"
	
	public FlagOption endOfWarningAfterWarningZoneOption = new FlagOption("endOfWarningAfterWarningZone", 'E', 
			"If this is ticked, for drift detectors that support warning zones, NA's warning window will finish simultaneously with the detector's warning zone.");

	public FlagOption alwaysResetWarningAtEarlyDriftOption = new FlagOption("alwaysResetWarningAtEarlyDriftOption", 'Z', 
			"If this is ticked, at any drift with warning window smaller than the treshold, the drift will be to a new blank classifier.");

	
	public FlagOption trainAfterDriftEvaluationOption = new FlagOption("trainAfterDriftEvaluation", 'B',
			"If enabled, it evaluates drifts as a predictive performance decrease (examples will not be fed to learner before drift detection).");

	public FlagOption testWithBKGOnWarnOption = new FlagOption("testWithBKGOnWarn", 'K',
			"If enabled, background learners will be used for testing rather than active ones during warnings (suitable for very sharp drifts).");
	
	public FlagOption insertOnEarlyDriftOption = new FlagOption("insertOnEarlyDrift", 'J',
			"If enabled, the active classifier will be inserted in case of early drift.");
	
	public FlagOption weightedTestsOnWarnOption = new FlagOption("weightedTestsOnWarn", 'L',
			"If enabled, both active and background learners will be tested (using a weighting mechanism).");
	
	public FloatOption betaOption = new FloatOption("beta", 'C', "Factor to punish mistakes by.", 0.5, 0.0, 1.0);

	public FlagOption trainActiveOnWarnOption = new FlagOption("trainActiveOnWarn", 'T',
			"If enabled, the active classifier is trained during warning too (this was enabled in the initial version of NA)");

	public FlagOption forceEarlyDriftsOption = new FlagOption("forceEarlyDrifts", 'F',
			"If enabled, it forces background drifts at early (sharpest) drifts (type 1 only: when the warning window is too small).");

	
	// public FloatOption stopPercentageOption = new FloatOption("stopPercentageOption", 'P',
	//		"Stopping criteria as percentage (if 0, the static stopping criteria is )", 0, 0, 100.0);

	//		IntOption("minTopologySizeForDrift", 'k',"Minimum number of prototypes created before allowing a drift.", 1, 1, Integer.MAX_VALUE);
	//////////
	
	protected NABaseLearner active;
	protected long instancesSeen;
	protected int subspaceSize;
	
	// Window statistics
	protected double lastError;

	// Warning and Drifts
	public long lastDriftOn;
	public long lastWarningOn;
	public long lastDriftDelta;

	// Drift and warning detection
	protected NAChangeDetector driftDetectionMethod;
//	protected ChangeDetector warningDetectionMethod;

	protected int numberOfDriftsDetected;
	protected int numberOfWarningsDetected;

	PrintWriter eventsLogFile;
	public int logLevel;

	Topology topology;
	Topology newTopology;
	Integer [] topologyAttributeSubset;
	Instances W;
	ConceptHistory CH;
	int conceptsSeen;
	int groupsSeen;
	boolean isOnWarningWindow;  // TODO: this flag could be removed, as the size of W is enough to know this.
	
	protected boolean debug_internalev = false;
	
	///////////////////////////////////////
	//
	// TRAINING AND TESTING OF THE ENSEMBLE
	// Data Management and Prediction modules are here.
	// All other modules also are orchestrated from here.
	// -----------------------------------
	
	@Override
	public void resetLearningImpl() {
		
		// Reset attributes
		this.active = null;
		this.subspaceSize = 0;
		this.instancesSeen = 0;
		this.topology = null;
		this.newTopology = null;
		this.conceptsSeen = 0;
		this.groupsSeen = 0;

		// Reset warning and drift detection related attributes
		this.lastDriftOn = 0;
		this.lastWarningOn = 0;
		this.lastDriftDelta = 0;

		this.numberOfDriftsDetected = 0;
		this.numberOfWarningsDetected = 0;

		// Init Drift Detector
		if (!this.disableDriftDetectionOption.isSet()) {
			this.driftDetectionMethod = new NAChangeDetector(this.driftDetectionMethodOption, 
					((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy());
		}
		// Init Drift Detector for Warning detection.
//		this.warningDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.warningDetectionMethodOption)).copy();
		this.W = new Instances();  // list of training examples during warning window.
	}

	/**
	 * In NA, this method performs the actions of the classifier manager. Thus, in
	 * this method, warning and drift detection are performed. This method also send
	 * instances to the ensemble classifiers. Or to the single active classifier if
	 * ensemble size = 1 (default).
	 *
	 * New BKG classifiers and switching from and to CH may also need to be here.
	 *
	 * This also trains base classifiers, track warning and drifts, and orchestrate the comparisons and replacement of classifiers.
	 * Drifts and warnings are only accessible from here.
	 *
	 * 	Steps followed:
	 *  ----------------
	 * - 0 Initialization
	 * - 1 If the concept history is ready and it contains old classifiers, then test the training instance
	 * 		in each old classifier's internal evaluator to know how their errors compare against bkg one.
	 * - 2 Update error in active classifier.
	 * - 3 Update error in background classifier's internal evaluator.
	 * - 4 Train each base classifier, orchestrating drifts and switching of classifiers.
	 * - 5 Train base classifier (Lines 4-6 of algorithm)
	 * - 6 Check for drifts and warnings only if drift detection is enabled
	 * - 7.1 Check for warning only if the warning window is active.
	 * - 7.1.1 Otherwise update the topology (this is done as long as there is no active warnings).
	 * - Step 7.2 Update warning detection adding latest error
	 * - Step 7.2.0 Check if there was a change (warning signal). If so, start warning window;
	 * - Step 7.2.1 Update the warning detection object for the current object.
	 * 			  This effectively resets changes made to the object while it was still a bkglearner.
	 * - Step 7.2.2 Start warning window.
	 * - 7.3 Check for drift
	 * - 8: Log training event
	 *
	 * The method below implements the following lines of the algorithm:
	 * - Line 1: start topology
	 * - Lines 2-3: initialize ensemble (create base classifiers) and lists.
	 * The rest of the lines of the algorithm are triggered from here
	 * - Line 4-6: ClassifierTrain(c, x, y) -> // Train c on the current instance (x, y).
	 * - Lines 14-15 if warning detection is disabled, update this topology
	 * - Line 16-19: If a warning is detected, start warning window and clear buffer W.	 *
	 * - Lines 7-20: warningDetection:
	 * - Lines 22-33: driftDetection:
	 */
	@Override
	public void trainOnInstanceImpl(Instance instance) {
		++this.instancesSeen;
//		System.out.println("ture class is -> " + (int) instance.classValue());
//		System.out.println("predicted label before training is -> " + Utils.maxIndex(this.active.getVotesForInstance(instance)));
		
//	   	if(this.instancesSeen > MIX_LIMIT_PRINTS & this.instancesSeen < MAX_LIMIT_PRINTS)
//    		System.out.println(this.instancesSeen);
//		System.out.println("EXAMPLES: " + this.instancesSeen);
		
        if(this.active == null || (!this.disableDriftDetectionOption.isSet() && this.topology == null))
    		init(instance); // Step 0: Initialization
//        System.out.println();
//        System.out.println(this.CH.classifiersKeySet());
//        System.out.println();
//                
//        System.out.println("instance: "+instancesSeen);
		// Step 1: Update error in concept history learners
		if (!this.disableRecurringDriftDetectionOption.isSet() && this.CH != null
				&& this.CH.getWarnings().containsValue(true) && this.CH.size() > 0) {
			this.CH.updateHistoryErrors(instance);
		} // Steps 2-4: Iterate through the ensemble for following steps (active and bkg classifiers)
		updateEvaluators(instance);

//		// Step 5: Train base classifier (Lines 4-6)
		if (!this.trainAfterDriftEvaluationOption.isSet()) {
			if (this.trainActiveOnWarnOption.isSet() || !this.isOnWarningWindow) // added for a test on 04/10/2020 (for tests that are trainSTART and not train Active during warning (0*0))
				this.active.trainOnInstance(instance); // for 0**
			// moved to the end by ASC on 26/09/2020
		}
		// System.out.print("before: ");
		// this.CH.printTopologies();  // debug
		// System.out.println("Instances seen: " + this.instancesSeen);  // debug
		// System.out.println("this top size:" + this.topology.getNumberOfPrototypesCreated());
		
		// Step 6: Check for drifts and warnings only if drift detection is enabled
		if (!this.disableDriftDetectionOption.isSet()) {
			boolean correctlyClassified = Utils.maxIndex(this.active.getVotesForInstance(instance)) == (int) instance.classValue();
			if (this.isOnWarningWindow) { // If on warning, update weight of the bkg learner based on its result
				// TODO: move to the end once I refactor the code to be GroCH only // aduarez 26/04/2021
				this.active.updateWeight(correctlyClassified, this.betaOption.getValue());
			    this.active.bkgLearner.updateWeight(Utils.maxIndex(this.active.bkgLearner.getVotesForInstance(instance)) == 
			    									(int) instance.classValue(), this.betaOption.getValue());
//			    System.out.println("weight of active: "+this.active.weight+"     ---      weight of bkg: "+this.active.bkgLearner.weight);
			}
			if (!this.trainAfterDriftEvaluationOption.isSet()) {
				if (this.isOnWarningWindow) trainDuringWarningImpl(instance); // Step 7.1: update actions on warning window   // for 0**
				else this.topology.trainOnInstanceImpl(instance); // Step 7.1.1: Lines 14-15: otherwise train topology  // for 0**
	////			System.out.println(this.instancesSeen+" "+this.isOnWarningWindow);
				// moved to the end by ASC on 26/09/2020
			}
			// Update the detection method (common for drift and warning)
			this.driftDetectionMethod.input(correctlyClassified ? 0.0 : 1.0); 
			if (this.driftDetectionMethod.getWarning()) { 
//	        	if(this.instancesSeen > MIN_LIMIT_PRINTS & this.instancesSeen < MAX_LIMIT_PRINTS)
//	        		System.out.println("WARNING ZONE!!");
//				System.out.println("INSTANCE: "+this.instancesSeen);
				// If a new warning is detected, GroCH starts its warning window. 
				// Apart from the warning zone (if the flag is active, the warning window will have at least and as maximum a duration of n examples, specified as an input parameter. 
				// This gives the GroCH a chance to foresee drifts preceded by blips or small gradual change periods).
				assert this.driftDetectionMethod.getWarningZone() == true; // otherwise there is a bug
				initWarningWindow(); // Step 7.2.2: new warning?
			}
			// Check if there was a change: line 22-23 drift detected?
//			// this.driftDetectionMethod.input(correctlyClassified ? 0 : 1);  // updated already
//			System.out.println(this.driftDetectionMethod.getChange());
			if (this.driftDetectionMethod.getChange() | ((this.instancesSeen - this.lastDriftOn) % this.periodBetweenDriftsOption.getValue() == 0)) {
//	        	if(this.instancesSeen > MIN_LIMIT_PRINTS & this.instancesSeen < MAX_LIMIT_PRINTS)
//	        		System.out.println("Enters in drift get change!");	        	
				long startTime = System.nanoTime();
				System.out.println("Lenght of warning window: " + this.W.size());
				this.lastDriftDelta = this.instancesSeen - this.lastDriftOn;
				System.out.println("Last drift was " + this.lastDriftDelta + " examples before.");
				System.out.println("DRIFT Delta equal or greater than insertion threshold? " + 
									(this.lastDriftDelta >= this.insertionPeriodOption.getValue()));
				driftHandlingImpl(correctlyClassified, instance); // Step 7.3: new drift?  (remove second param sent if early drifts dont need to be forced)
				if (this.eventsLogFile != null && this.logLevelOption.getValue() >= 1) {
					long endTime = System.nanoTime();
					logEvent(prepareDriftDurationEvent((endTime - startTime) / 1000000)); // in milliseconds.
				}
			}
		}
		
		// Step 5: Train base classifier (Lines 4-6)
		if (this.trainAfterDriftEvaluationOption.isSet()) {
			if (this.trainActiveOnWarnOption.isSet()) this.active.trainOnInstance(instance); // moved to only train outside warning on 30/09/2020
			if (this.isOnWarningWindow) trainDuringWarningImpl(instance); // Step 7.1: update actions on warning window 
			else {
				if (!this.trainActiveOnWarnOption.isSet()) this.active.trainOnInstance(instance);	// moved on 30/09/2020
				this.topology.trainOnInstanceImpl(instance); // Step 7.1.1: Lines 14-15: otherwise train topology
			}
		}
		
		// Step 8: Register training example in log
		if (this.eventsLogFile != null && this.logLevelOption.getValue() >= 2)
			logEvent(getTrainExampleEvent());
	}

	public void updateEvaluators(Instance instance) {
//		DoubleVector pred = new DoubleVector(this.active.getVotesForInstance(instance));
		double[] pred = this.active.getVotesForInstance(instance);
		InstanceExample example = new InstanceExample(instance);
		this.active.evaluator.addResult(example, pred); //vote.getArrayRef()); // Step 2: Testing in active classifier

		if (!disableRecurringDriftDetectionOption.isSet()) { // Step 3: Update error in background classifier's
			if (this.active.bkgLearner != null && this.active.bkgLearner.internalWindowEvaluator != null
					&& this.active.bkgLearner.internalWindowEvaluator
							.containsIndex(this.active.bkgLearner.indexOriginal)) {
//				DoubleVector bkgVote = new DoubleVector(this.active.bkgLearner.getVotesForInstance(instance));
				double[] bkgPred = this.active.bkgLearner.getVotesForInstance(instance);
				
				// Update both active and bkg classifier internal evaluators
				this.active.bkgLearner.internalWindowEvaluator.addResult(example, bkgPred); // bkgVote.getArrayRef());
				this.active.internalWindowEvaluator.addResult(example, pred); // vote.getArrayRef());
			}
		}
	}
	

	@Override
	public double[] getVotesForInstance(Instance instance) { // although just one active learner. so only this one votes. legacy code from RCARF
		Instance testInstance = instance.copy();
		if (this.active == null) init(testInstance);
//		double[] pred = this.active.getVotesForInstance(testInstance);
//    	if(this.instancesSeen > MIN_LIMIT_PRINTS & this.instancesSeen < MAX_LIMIT_PRINTS) {
////    		System.out.println("instance is: "+testInstance.toString());
//    		System.out.println("prediction on instance: "+this.instancesSeen+"  "+pred[0]+"/"+pred[1]);
//    	}
//		return pred;
		if (this.weightedTestsOnWarnOption.isSet()) {
			// If the bkg learner has more weight than the active learner, the prediction is performed with this.
			if (this.W.numInstances() >= this.minWSizeForDriftOption.getValue() &&
					this.active.bkgLearner.weight > this.active.weight) // only during the warning window
				return this.active.bkgLearner.getVotesForInstance(testInstance);
			else return this.active.getVotesForInstance(testInstance);
		}
		else {
			if (this.testWithBKGOnWarnOption.isSet()){
				if (this.W.numInstances() >= this.minWSizeForDriftOption.getValue()) {
		//			System.out.println("Testing with BKG learner during WW.");
					return this.active.bkgLearner.getVotesForInstance(testInstance);
				}
			} return this.active.getVotesForInstance(testInstance);
		}
	}

	@Override
	public boolean isRandomizable() {
		return true;
	}

	@Override
	public void getModelDescription(StringBuilder arg0, int arg1) {
	}

	@Override
    protected Measurement[] getModelMeasurementsImpl() {
        List<Measurement> measurementList = new ArrayList<Measurement>();
        measurementList.add(new Measurement("Change detected", this.numberOfDriftsDetected));
        measurementList.add(new Measurement("Warning detected", this.numberOfWarningsDetected));
        // TODO. add all columns from logEvents object, to remove this object
        this.numberOfDriftsDetected = 0;
        this.numberOfWarningsDetected = 0;
        return measurementList.toArray(new Measurement[measurementList.size()]);
    }
	
	private void init(Instance instance) {
		
		// If window resize policy -1 (always growing), then ...
		if (this.windowResizePolicyOption.getValue() == -1) {
			this.disableResizingOfAllWindowsOption.unset(); // GroCH does so for the CH classifiers as well
			this.minWindowSizeOption.setValue(1); // the minimum is == 1 (when the warning window starts)
			this.defaultWindowOption.setValue(1); // the default is == 1 (it starts at one when the warning window starts), and keeps growing.
		}
		// Only initialize the Concept History if recurring concepts are enabled
		if (!this.disableRecurringDriftDetectionOption.isSet() && this.CH == null) {
					
			// Parse attribute subset for clustering if this is applied
			String [] featSubsetUnparsed = this.topologyAttributeSubsetOption.getValue().split(",");
			if (featSubsetUnparsed.length > 1 && featSubsetUnparsed[0] != "-1") {
				this.topologyAttributeSubset= new Integer [featSubsetUnparsed.length]; 
				for (int i = 0; i < featSubsetUnparsed.length; i++) 
					this.topologyAttributeSubset[i] = Integer.parseInt(featSubsetUnparsed[i]);
			}
			String [] pretrainSets = this.arffToInitCHOption.getValue().split(";");
//			if (pretrainSets[0] == null || pretrainSets[0] == "-1") {
//				pretrainSets = new String[1]; 
//				// TODO: if no pretraining datasets are specified, get first rows from current one
				// 1. exclude n rows (= this.rowsToPreTrainMatrixOption.getValue()) 
				//		from the current dataset (exclude them from test and train and test that this works properly)
				// 2. pretrainSets[0] = n_first_rows_current_dataset/stream/instances-obj
//			}
			this.CH = new ConceptHistory(this.distThresholdOption.getValue(), 
										 this.topologyAttributeSubset, 
										 this.distanceMetricOption.getChosenLabel(), 
										 this.numberOfMatricesOption.getChosenLabel(),
										 pretrainSets);   // pretraining sets for Mahalanobis distance matrix need to be specified manually by now
			if (disableTopologyLearnerOption.isSet()) this.topology = new Topology(this.topologyAttributeSubset, this.trainOnlineFlagOption.isSet());
			else {
				if (this.CH.computeSingleMatrix) this.topology = new Topology(this.topologyLearnerOption, 
						this.topologyAttributeSubset, this.CH.m_dist, this.trainOnlineFlagOption.isSet()); 
				else this.topology = new Topology(this.topologyLearnerOption, this.topologyAttributeSubset, this.trainOnlineFlagOption.isSet()); 
			}
			// algorithm line 1
			this.topology.resetLearningImpl();
			if (this.initializeConceptHistoryOption.isSet()) initConceptHistory();

		}
		if (this.active == null) { // algorithm lines 2-3
			// Init the ensemble.
			BasicClassificationPerformanceEvaluator classificationEvaluator = (BasicClassificationPerformanceEvaluator)
			      getPreparedClassOption(this.evaluatorOption);

			this.W = (Instances) instance.copy().dataset();
			this.W.delete();
			this.isOnWarningWindow = false;

			// START: TO BE REMOVED ONCE THE LOGS ARE NO LONGER REQUIRED
			try { // Start events logging and print headers
				if (this.disableEventsLogFileOption.isSet()) {
					this.eventsLogFile = null;
				} else {
					this.eventsLogFile = new PrintWriter(this.eventsLogFileOption.getValue());
					logEvent(getEventHeaders());
				}
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} // END -TO BE REMOVED

			Classifier learner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
			learner.resetLearning();

			this.active = new NABaseLearner(0, // active classifier pos in an ensemble of active classifiers
					(Classifier) learner.copy(),
					(BasicClassificationPerformanceEvaluator) classificationEvaluator.copy(), this.instancesSeen,
					!this.disableDriftDetectionOption.isSet(), // these are still needed con the level below
					false, // is bkg?
					!this.disableRecurringDriftDetectionOption.isSet(),
					false, // first classifier is not in the CH.
					new Window(this.defaultWindowOption.getValue(),
							1, // size of increments/decrements
							this.minWindowSizeOption.getValue(),
							0.65, // "Decision threshold for recurring concepts (-1 = threshold option disabled)."
							false, // rememberConceptWindowOption?
							!this.disableResizingOfAllWindowsOption.isSet() ? true : false,
							this.windowResizePolicyOption.getValue()), // "Policy to update the size of the window. 0 in RCARF."
					null, // internalEvaluator window starts as NULL
					1, // first active learner is being used already once
					0); // no trained examples yet
		}
	}
	
	
	///////////////////////////////////////
	//
	// CLASSIFIER MANAGEMENT MODULE
	// Divided into three parts:
	// - Warning and drift handling (detection)
	// - Actions in case of drift
	// - Identification of new best group, classifier and trigger switch between classifiers.
	// -----------------------------------
	
	// WARNING AND DRIFT HANDLING

	/**
	 * This method implements all the actions that happen when the warning detection is enabled.
	 * ---------------------------------------------------------------------
	 * Some of the following lines of the algorithm GroCH are implemented here:
	 * - Lines 7-10: Disable warnings after a certain period of time. False warnings handling at buffer W level
	 * - Line 11: if (size(W) ð��� (ð�Ÿ�, ð���)) ->In warning window
	 * - Line 12: Train the background classifier
	 * - Line 13: Add instance to the buffer of instances during warning
	 *
	 * The steps followed for this can be seen below:
	 * - Step 1 Disable warning if a length threshold is reached - Lines 7-11
	 * - Step 2 If the classifier is in the warning window, train the bkg classifier and add the current instance to W.
	 */
	protected void trainDuringWarningImpl(Instance inst) {
		// Step 1: Check if the warning window should be disabled (Lines 7-10)
		// TODO: we could also disable warnings by using a signal to noise ratio variation of the overall classifier during warning
		if (this.W.numInstances() > this.warningWindowSizeThresholdOption.getValue() || 
				// If the detector supports warning zones, drifts should be in the warning zone.
				(this.endOfWarningAfterWarningZoneOption.isSet() && 
				!this.driftDetectionMethodOption.getValueAsCLIString().contains("ADDM") && // TODO: create a list/enum or something listing detectors that do not support getWarningZone()
				!this.driftDetectionMethod.getWarningZone())) 
			resetWarningWindow(true); // Line 8  // this is an addition when comparing to other state of the art classifiers under drift, as RCD and CPF
		else { 		 	
			// Step 2: Either warning window training/buffering or topology update (Lines 11-15)
			this.active.bkgLearner.classifier.trainOnInstance(inst); // Line 11
			this.W.add(inst.copy()); // Line 12
			
			// DEBUG
			if (this.debug_internalev) {
//				System.out.println("------------------");
				System.out.println("W size: "+this.W.size());
//				System.out.println("Internal window size: "+this.active.internalWindowEvaluator.getCurrentSize(0));
//				System.out.println("BKG Internal window size: "+this.active.bkgLearner.internalWindowEvaluator.getCurrentSize(0));
//				System.out.println("------------------");
			}
		}
		// System.out.println("SIZE OF W: "+ W.numInstances());
	}
	
	/**
	 * The boolean refers to the start of a warning window
	 * */
	protected void resetWarningWindow(boolean startingWW){
		if (startingWW) {
			this.active.bkgLearner = null; // Lines 8 and 18
			this.active.internalWindowEvaluator = null;
			this.active.tmpCopyOfClassifier = null;
			System.out.println("[Example " + this.instancesSeen + "] BKG AND TMP CLASSIFIERS AND INTERNAL WINDOW RESTARTED");
			// Change 3 on 31/10/2020: if we reset the warning, we should train the topology with the last received examples. 
			//  we should also train the active learner if it has not been trained during warning, to avoid hundreds of pointless drifts/warnings.
			
//			System.out.println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
//			System.out.println("WARNING WINDOW HAS " + this.W.size() + " EXAMPLES THAT WILL BE FED TO THE TOPOLOGY");
			for (int i = 0; i < this.W.size(); i++) {
				Instance inst = this.W.get(i).copy();
				if (!this.trainActiveOnWarnOption.isSet()) this.active.classifier.trainOnInstance(inst);
				this.topology.trainOnInstanceImpl(inst);
			}
			// this below trains GNG with the W if relevant.
//			if (!this.trainOnlineFlagOption.isSet()) this.topology.trainFromBuffer();  // commented out cos in this case the topology won't be inserted and it's pointless to train it

//			System.out.println("END OF TRAINING DURING RESET!");
//			System.out.println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
//			System.out.println("Topology after drift has seen " + this.topology.getInstancesSeenByLearner() + " examples");
//			System.out.println("Base classifier after drift has seen " + this.active.instancesTrained + " examples");
//			System.out.println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
		}
		((NAChangeDetector) this.driftDetectionMethod).resetWarningDetector();  // TODO: check ASC 21/08/2020
		
//		System.out.println("W size was: "+W.numInstances());
		// TODO: W and isOnWarningWindow params could be embedded/merged with NAChangeDetector (isOnWarningWindow should be == warningZone)
		this.W.delete(); // Lines 9 and 19 (it also initializes the object W)
		System.out.println("W ERASED!");
		this.isOnWarningWindow = false;
		if (!this.disableRecurringDriftDetectionOption.isSet()) this.CH.decreaseNumberOfWarnings(0); // update applicable concepts
	}
	
	/**
	 * This starts the warning window event
	 *
	 * The next steps are followed:
	 * - 0 Reset warning
	 * - 1 Update last error and make a backup of the current classifier in a concept object
	 * 		(the active one will be in use until the Drift is confirmed).
	 * - 2 Update of objects with warning.
	 * - 3 If the concept internal evaluator has been initialized for any other classifier on warning,
	 * 		add window size and last error of current classifier on warning.
	 *		Otherwise, initialize a new internal evaluator for the concept
	 * */
	protected void initWarningWindow() {
		this.lastWarningOn = this.instancesSeen;
		this.numberOfWarningsDetected++;
		
		// Step 0 (Lines 17-19)
		resetWarningWindow(true);

		// Step 1 Update last error and make a backup of the current classifier
		if (!this.disableRecurringDriftDetectionOption.isSet()) {
			this.active.saveCurrentConcept(this.instancesSeen); // line 18: Save a tmp copy of c as snapshot
		}
		// Step 2: Update of objects with warning.
		if (!this.disableRecurringDriftDetectionOption.isSet()) {
			this.active.createInternalEvaluator();
			this.CH.increaseNumberOfWarnings(0, this.active, this.lastError); // 0 is always the pos in an ensemble of only 1 active learner
		}
		if (this.eventsLogFile != null && this.logLevelOption.getValue() >= 1) logEvent(getWarningEvent()); // Log this

		// Step 3: Create background Classifier
		this.active.createBkgClassifier(this.lastWarningOn); // line 19: create background classifier
		this.isOnWarningWindow = true;
		
		// Step 4: Reset weights to start even at the start of the warning window.
		this.active.resetWeight();
		this.active.bkgLearner.resetWeight();
	}

	protected void initConceptHistory() {
		
		// Load datasets into an array
		String [] paths = this.arffToInitCHOption.getValue().split(";");	
		if(paths[0] != "-1") {
			Instances [] datasets = new Instances[paths.length]; 
			for (int i = 0; i < datasets.length; i++) datasets[i] = loadDataset(paths[i]);
			
		    // for loop through 4 datasets
			for (int datasetID = 0; datasetID < datasets.length; datasetID++) {
				// 1 Create topology to GNG.
				Topology top = this.topology.clone(); // line 30  (new Topology does not create a new object due to the meta class for the clusterer)
				top.resetId();
				top.resetLearningImpl(); // line 30
				
				// 2 Create classifier
				Classifier c = (Classifier) getPreparedClassOption(this.baseLearnerOption);
				c.resetLearning();
				BasicClassificationPerformanceEvaluator classificationEvaluator = (BasicClassificationPerformanceEvaluator)
				      getPreparedClassOption(this.evaluatorOption);
				NABaseLearner learner = new NABaseLearner(0, // active classifier pos in an ensemble of active classifiers
						(Classifier) c.copy(),
						(BasicClassificationPerformanceEvaluator) classificationEvaluator.copy(), this.instancesSeen,
						!this.disableDriftDetectionOption.isSet(), // these are still needed con the level below
						false, // is bkg?
						!this.disableRecurringDriftDetectionOption.isSet(),
						false, // first classifier is not in the CH.
						new Window(this.defaultWindowOption.getValue(),
								1, // size of increments/decrements
								this.minWindowSizeOption.getValue(),
								0.65, // "Decision threshold for recurring concepts (-1 = threshold option disabled)."
								false, // rememberConceptWindowOption?
								!this.disableResizingOfAllWindowsOption.isSet() ? true : false,
								this.windowResizePolicyOption.getValue()), // "Policy to update the size of the window. 0 in RCARF."
						null, // internalEvaluator window starts as NULL
						0,
						0); // these learners pre-trained have never been used
				
				// 3 Instances to classifier and GNG
				int instancesSeen = 0;
				for (; instancesSeen < datasets[datasetID].numInstances(); instancesSeen++) {
					Instance inst = datasets[datasetID].get(instancesSeen);
					// add to learner
//					learner.getVotesForInstance(inst);  // only to use the evaluator (it may not be necessary) TODO: remove?
					learner.trainOnInstance(inst); 
					
					// add to GNG
					top.trainOnInstanceImpl(inst);
				}	
				if(!this.trainOnlineFlagOption.isSet()) top.trainFromBuffer();
				if(!this.insertExamplesOnlyOption.isSet()) learner.resetConceptBuffer();

				// 4 Add a new group to CH and a concept containing the classifier to that group
				int previousTopologyGroupId = this.groupsSeen++;
				Concept cn = new Concept(learner, 0, 0.5, instancesSeen);
				cn.setHistoryID(this.conceptsSeen++);				
				this.CH.createNewGroup(previousTopologyGroupId, top.clone(), this.topologyAttributeSubset); 
				this.CH.addLearnerToGroup(previousTopologyGroupId, cn);	
				
				// Debugging
				System.out.println("=========================");
				System.out.println("Creating group #"+datasetID+" with ID: "+previousTopologyGroupId+
						" Trained with "+instancesSeen+" instances from file "+paths[datasetID]);
				System.out.println("Its topology has "+(this.CH.get(previousTopologyGroupId).getTopologyPrototypes().numAttributes())+" prototypes.");
				System.out.println("It has now "+(this.CH.get(previousTopologyGroupId).values()).size()+" concepts");
//				System.out.println("Its classifier evaluator has a performance of: " + classificationEvaluator.getPerformanceMeasurements()[0].getValue());
			}	
			logMeanDistancesBetweenPretrainedGroups();
		}		
	}

	private void logMeanDistancesBetweenPretrainedGroups() {
		// START of part for reporting purposes only
		// 11-01-2021 ASC
		System.out.println("Computing distance between groups");
		for (int i = 0; i < this.CH.size(); i++) {
			for (int j = 0; j < this.CH.size(); j++) {
//				if ( i != j ) {
					double dist = 0.0;
					// Mahalanobis will consider classes as any other attribute for the distances 
					// Euclidean will in computeDistances(), but it may not in LinearNNSearch. TODO: check and fix if required
					if (this.CH.distanceMetric.equals("Euclidean")) {
						dist = this.CH.getMeanDistanceToNN(this.CH.get(i).getTopologyPrototypes(), 
								this.CH.get(j).getTopologyPrototypes()); 
						System.out.printf("Mean Euclidean distance between groups "+i+" and "+j+" is %.3f", dist);

					} else if (this.CH.distanceMetric.equals("Mahalanobis")) {
						if (this.CH.computeSingleMatrix) {
							dist = this.CH.m_dist.distance(this.CH.get(i).getTopologyPrototypes(), 
									this.CH.get(j).getTopologyPrototypes());
						} else {
							System.out.println();
							SamoaToWekaInstanceConverter converter = new SamoaToWekaInstanceConverter();
							MahalanobisDistance m = new MahalanobisDistance();  // TODO: move this up
							dist = m.distance(converter.wekaInstances(this.CH.get(i).getTopologyPrototypes()), 
										      converter.wekaInstances(this.CH.get(j).getTopologyPrototypes()));
						} // Debugging
						System.out.printf("Mean Mahalanobis distance between groups "+i+
								" (size: "+this.CH.get(i).getTopologyPrototypes().size()+") and "+j+
								" (size: "+this.CH.get(j).getTopologyPrototypes().size()+") is %.3f", dist);
						System.out.println();

					} else {
						System.out.println("Please, specify a distance metric for concept similarity.");
					}
//				}
				
			}
		}
		// END of part for reporting purposes only
	}

	
	protected Instances loadDataset(String pathToArff) {
		WekaToSamoaInstanceConverter conv = new WekaToSamoaInstanceConverter();
		BufferedReader reader;
		Instances data = null;
		try {
			reader = new BufferedReader(new FileReader(pathToArff));
			ArffReader arff = new ArffReader(reader);
			weka.core.Instances staticDataset = arff.getData();
			staticDataset.setClassIndex(staticDataset.numAttributes() - 1);
			data = conv.samoaInstances(arff.getData());
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} return data;
	}
	
	
	/**
	 * Class that defines the properties of the drift type when switching the active classifier
	 * */
	public class ConceptDrift {
		
		protected boolean falseAlarm;
		protected boolean bkgDrift;
		
		// Parameters regarding the retrieval
		protected boolean activeImprovesRetrieved;
		protected HashMap<Integer, Double> rankingOfRetGroupClassifiers;
		
		// Unused params
		protected int indexOfBestRecurringClassifier;
		protected int groupRetrieved;
		
		public ConceptDrift() {
			this.falseAlarm = false;
			this.activeImprovesRetrieved = false;
			this.rankingOfRetGroupClassifiers = null;
			this.indexOfBestRecurringClassifier = -1;
			this.groupRetrieved = -1;
			this.bkgDrift = false;
		}
		
		public ConceptDrift(boolean falseAlarm, boolean activeImprovesRetrieved, 
				HashMap<Integer, Double> rankingOfRetGroupClassifiers, 
				int indexOfBestRanked, int groupRetrieved, boolean bkgDrift){
			this.falseAlarm = falseAlarm;
			this.activeImprovesRetrieved = activeImprovesRetrieved;
			this.rankingOfRetGroupClassifiers = rankingOfRetGroupClassifiers;
			this.indexOfBestRecurringClassifier = indexOfBestRanked;
			this.groupRetrieved = groupRetrieved;
			this.bkgDrift = bkgDrift;
		}
		
		public boolean isFalseAlarm() {
			return this.falseAlarm;
		} 
		
		public boolean isBkgDrift() {
			return this.bkgDrift;
		}
		
		public boolean doesActiveImproveRecurring() {
			return this.activeImprovesRetrieved;
		}

		public int getRetrievedGroupID() {
			return this.groupRetrieved;
		}
		
		public void setFalseAlarm() {
			this.falseAlarm = true;
		}
	
	} 
	
	/**
	 * This method selects the next concept classifier and closest group topology when a drift is raised.
	 * 	Pselected is: a new P (Pn) in case of bkgDrift; Pc in case of false alarm; and Ph in case of recurring drift.
	 *
	 * The next steps are followed:
	 * - 0 Set false in case of drift at false as default.
	 *     Included for cases where driftDecisionMechanism > 0 and recurring drifts are enabled.
	 * - 1 Compare DT results using Window method and pick the best one between CH and bkg classifier.
	 *     It returns the best classifier in the object of the bkgLearner if there is not another base classifier
	 *      with lower error than active classifier (and driftDecisionMechanism > 0), then a false alarm is raised.
	 *     This step belong to line 29 in the algorithm: c = FindClassifier(c, b, GH) -> Assign best transition to next state.
	 * - 2 Orchestrate all the actions if the drift is confirmed and there is not a false alarm.
	 * - 4 Decrease amount of warnings in concept history and from evaluators
	 * - 3 reset base learner
	 *
	 * Lines of the algorithm Lines 22-25 are implemented here:
	 * -----------
	 * Insertion in CH (Lines 24-28)
	 * 	line 24: get prototypes from topology
	 * 	line 25: Group for storing old state
	 * 	line 26-27: create a new group represented by 'tmpPrototypes' from Tc
	 * Retrieval from CH and refresh (lines 28-32)
	 * 	line 28: push current classifier to Gc
	 * 	line 29: Update topology on Gc
	 * 	line 30: Group for retrieval of next state (selectDrift()) - In method switchActiveClassifier
	 * 	line 31: Add the examples during warning to a new topology.
	 * 	line 32: reset list W and warning flag
	 */
	protected void driftHandlingImpl(boolean correctlyClassified, Instance inst) {  // (remove second param sent if early drifts dont need to be forced)
		this.lastDriftOn = this.instancesSeen;
		this.numberOfDriftsDetected++;
		ArrayList<Integer> groupsRetrieved = new ArrayList<Integer>();
		ConceptDrift switchInfo = new ConceptDrift(); // Set false alarms (case 1) at false as default
		assert this.newTopology == null;  // it should start this as null

		// 1st Train Topology from buffer if the training is not online
		if (this.W.numInstances() >= this.minWSizeForDriftOption.getValue()) {
			System.out.println(); System.out.println(); System.out.println();
			System.out.println(); System.out.println(); System.out.println();
			System.out.println("DRIFT DETECTED!");
			
			// Retrieval from CH
			if (!this.disableRecurringDriftDetectionOption.isSet()) { // step 1: lines 29-31 of the algorithm
				// Start retrieval from CH (TODO: W should only be used if warning detection is enabled. same for topologies?)
			    if (this.CH.hasGroups()) groupsRetrieved = this.CH.findGroups(this.W, false, this.multiClusterOption.isSet());
			    switchInfo = switchActiveClassifier(groupsRetrieved);
			    if (this.eventsLogFile != null && this.logLevel >= 0) logEvent(getRetrievalFromCHEvent(switchInfo.getRetrievedGroupID()));  // this registers all retrievals (also in case of false alarms)
//				if (!falseAlarm) { if (this.eventsLogFile != null && this.logLevel >= 0) logEvent(getRetrievalFromCHEvent(retrievalGroupId));  }  // retrievals here only would consider drifts
			} else if (this.eventsLogFile != null && this.logLevelOption.getValue() >= 1) {  // TODO: remove this.eventsLogFile != null as a condition
				if(this.multiPassTopologyTrainingOption.isSet()) this.topology.trainUntilFulfillingStoppingCriteria();
				if (this.eventsLogFile != null && this.logLevel >= 0) logEvent(getBkgDriftEvent());  // TODO. Refactor this logEvent function so it's inside of 'registerDrift' and not wrapping it
			}
			if (!switchInfo.isFalseAlarm()) insertionAndDrift(switchInfo, 
					true & 
					(this.active.instancesTrained > this.minInsertionThresholdOption.getValue()) & 
					(this.lastDriftDelta < this.insertionPeriodOption.getValue())); // Step 2
			
		} else {
			// This forces Early drifts type 1 only (warning window small). This is done by forcing a BKG drift.
			// Early drifts type 2 (not enough prototypes) may corrupt the Concept History. Thus, they shouldn't be forces.
			if (this.forceEarlyDriftsOption.isSet()) {
					System.out.println("There weren't enough instances in W. "
							+ "In fact there were " + this.W.size() + " examples in W. \n"
							+ "A drift was forced on [Example " + this.instancesSeen + "] as a consequence. \n"
							+ "If this happens often, check the confidence intervals assigned for warnings vs drifts.");
					// If WW is 0, we initialize it to force the BKG drift.
					if (this.W.numInstances() < 1 | this.alwaysResetWarningAtEarlyDriftOption.isSet()) {
						initWarningWindow(); // a bkg learner and internal evaluators are created for comparison
						// TODO: should I enable this? onl when num instances < 1
						this.active.bkgLearner.classifier.trainOnInstance(inst); // train at least with 1
					}
					assert groupsRetrieved.size() == 0;
					groupsRetrieved.add(-1);
				    switchInfo = switchActiveClassifier(groupsRetrieved);
					if (!switchInfo.isFalseAlarm()) {
						insertionAndDrift(switchInfo, insertOnEarlyDriftOption.isSet()); // Step 2  (just drift but not insertion if its a false alarm)
					} // if it's a false alarm, it is registered already in switchActiveClassifier()
			}
			else {
				if(this.W.numInstances() < this.minWSizeForDriftOption.getValue())
					System.out.println("There weren't enough instances in W. In fact there were " + this.W.size() + " examples in W. "
							+ "If this happens often, check the confidence intervals assigned for warnings vs drifts.");
				registerDriftFalseAlarm();
				switchInfo.setFalseAlarm();
			}
		}
		if (switchInfo.isFalseAlarm() && this.saveClassifiersOnFalseAlarmOption.isSet() && this.active.tmpCopyOfClassifier != null) {
			System.out.println("IMP: There was a false alarm. But as its flag is true, the tmp copy of the classifier before warning is sent to the Concept History.");				
			insertConceptToCH(switchInfo); // save tmp copy of active classifier to CH as agreed with acervant and dquintan on 09/12/2019						
		}
//		assert groupRetrieved == switchInfo.getRetrievedGroupID();
		System.out.println("Pn size:" + this.topology.getNumberOfPrototypesCreated());
		// Reset drift independently on both false alarm and actual drift cases.
//		this.driftDetectionMethod = new NAChangeDetector(this.driftDetectionMethodOption, 
//				((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy());
		if (this.driftDetectionMethodOption.getValueAsCLIString().contains("ADDM")) this.driftDetectionMethod.resetDriftDetector(); // In detectors != ADWIN, this will also reset warnings as they're embedded in the classifier
		else this.driftDetectionMethod.setDriftDetector(((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy());

		this.newTopology = null;
		System.out.println("Although the new topology is null, the old one is of size: "+this.topology.getPrototypes().size());
		/*System.out.print("just after creation: ");
		this.CH.printTopologies();
		System.out.println();*/
	}

	protected void insertionAndDrift(ConceptDrift switchInfo, boolean insert) {
		if (insert) insertConceptToCH(switchInfo);  // Insertion in CH (Lines 23-26) 
		this.active.reset(); // reset base classifier and transition to bkg or recurring learner (step 3)
		
		System.out.println("OLD TOPOLOGY USED FOR INSERTION. NOW 'TRANSFER' OF EXAMPLES RECEIVED DURING WARNING.");
		// Reset warning related params (but not the learners as these are used at the next step)
		if (this.newTopology != null) this.topology = updateTopology(this.newTopology.clone(), this.W); // line 31
//		else {
//			this.topology = updateTopology(this.topology, this.W); // added on 15/01/2021		
//			System.out.println("ALERT: THE NEW TOPOLOGY WAS NULL");
//		}
		assert this.newTopology != null; // if this happens at any time, let's log it properly as above (commented out)
		// 26-01-2021: line below commented out  (so we can see if what decreases performance is to train the WW after drift) // TODO: add option?
//		if (!switchInfo.bkgDrift) for (int i = 0; i < this.W.size(); i++) this.active.trainOnInstance(this.W.get(i)); // NEW LINE 15-01-2021. // commented out on 31/01/2021. already covered by 'change 1' done at 31/10/2020.
		if (!this.trainOnlineFlagOption.isSet()) this.topology.trainFromBuffer();  // this trains GNG with the W if relevant.
		System.out.println("W HAS " + this.W.size() + " INSTANCES.");
		System.out.println("Topology after drift has a size of: " + this.topology.getPrototypes().size());
		System.out.println("Topology after drift has seen: " + this.topology.getInstancesSeenByLearner() + " examples. ");
		System.out.println("Base classifier after drift has seen: " + this.active.instancesTrained + " examples");
		resetWarningWindow(false); // step 4 and line 32. false as argument as the bkg related params are reseted in the prior line.
	}

	/**
	 * Method to handle insertions in the Concept History]
	 * @param groupRetrieved: if any group has been retrieved (id!=-1), then don't insert to the same group.
	 * @param activeImprovesBestRetrieved: only = True if the error of the active classifier in the internal evaluator 
	 * 			is lower than the error for the best retrieved classifier (if any group was retrieved)
	 */
	protected void insertConceptToCH(ConceptDrift retrievalProperties) {
		
		// Update topology for the insertion if the training of it it's done in batches when a drift is detected
		System.out.println("Training from buffer? "+ !this.trainOnlineFlagOption.isSet());
		if (!this.trainOnlineFlagOption.isSet()) this.topology.trainFromBuffer();
		
		System.out.println();
		System.out.println("Pc size:" + this.topology.getNumberOfPrototypesCreated());
		
		// Insertion in CH (Lines 23-26)
		if (this.topology.getNumberOfPrototypesCreated() >= this.minTopologySizeForDriftOption.getValue()){
			System.out.println("The topology has been trained (before the warning) with "+this.topology.getInstancesSeenByLearner()+" instances so far.");
			System.out.println("This is beyond the treshold so GroCH proceeds to insert to the CH.");
			
			// First, it needs to find the group ID for the insertion
			Instances tmpPrototypes = this.topology.getPrototypes(); // line 23
			int groupRetrieved = retrievalProperties.getRetrievedGroupID();
			boolean activeImprovesBestRetrieved = retrievalProperties.doesActiveImproveRecurring();
			int groupInserted = this.CH.findGroups(tmpPrototypes, true, false).get(0); // line 24
			if (this.eventsLogFile != null && this.logLevel >= 0) logEvent(getInsertionToCHEvent(groupInserted));
			if (this.CH.size() == 0 || groupInserted == -1) { // line 25
				groupInserted = this.groupsSeen++;
	//			System.out.println("CREATING NEW GROUP: " + previousTopologyGroupId); 
				this.CH.createNewGroup(groupInserted, this.topology.clone(), this.topologyAttributeSubset); //, this.newTopology); // line 27
			}
	//		else System.out.println("SELECTING FROM GROUP: " + previousTopologyGroupId); 
						
			// Move copy of active classifier made before warning to Concept History.
			this.active.tmpCopyOfClassifier.setHistoryID(this.conceptsSeen++);
			assert this.maxGroupSizeOption.getValue() == 1;  // please review all TODO's below for mas group size == 1. they need to be approached so the algorithm is ready
			
			// Only insert if the group differs
			if (groupRetrieved != groupInserted || activeImprovesBestRetrieved || 
					this.alwaysInsertClassifiersOption.isSet()) {  // TODO: alwaysInsertClassifiersOption not tested
				// we insert this first to take it also into account (so it's considered by the replacement policy)
	//			System.out.println("GROUP INSERTED - activeHasLowerErr == "+activeImprovesBestRetrieved);
	//			this.CH.addLearnerToGroup(groupInserted, this.active.tmpCopyOfClassifier); // line 28  // debug. check that this is not a duplicate. that it's a separate object.
				System.out.println("GROUP SIZE IS: "+this.CH.getConceptsFromGroup(groupInserted).size());
				if (this.CH.getConceptsFromGroup(groupInserted).size() >= maxGroupSizeOption.getValue()){
					System.out.println("GROUP MAX SIZE EXCEEDED!");
					// Then remove a classifier using a policy
					switch (this.groupReplacementPolicyOption.getChosenLabel()) {
	//					case "Greatest Error Out":
	//						assert groupInserted != -1;
	//						// 2 Retrieve best applicable classifier from Concept History (if a CH group applies)
	//						HashMap<Integer, Double> ranking = rankConceptHistoryClassifiers(groupInserted);
	//						if (ranking.size() > 0) {
	//							System.out.println("5555555 groups ranking for removal of classifiers before insertion just below:");
	//							System.out.println(ranking.toString());
	//							System.out.println("555555 TEST SIZE OF CH GROUP - 1 BEFORE REMOVAL: "+this.CH.getConceptKeySetFromGroup(groupInserted));
	//							this.CH.removeLearnerFromGroup(groupInserted, getMaxKey(ranking));
	//							System.out.println("555555 ranking now is (see below):");
	//							System.out.println(rankConceptHistoryClassifiers(groupInserted));
	//							System.out.println("555555 TEST SIZE OF CH GROUP - 2 AFTER REMOVAL: "+this.CH.getConceptKeySetFromGroup(groupInserted));
	//						}						
	//						break;
	//					case "LRU":   
							// This is done by FIFO now
	//						System.out.println("555555 TEST SIZE OF CH GROUP - 1 BEFORE REMOVAL: "+this.CH.getConceptKeySetFromGroup(groupInserted));
	//						System.out.println("555555 MODIFIED TIMES: "+this.CH.getConceptLastTimeUsedFromGroup(groupInserted).toString());
	//						this.CH.removeLearnerFromGroup(groupInserted, getMinKey(this.CH.getConceptLastTimeUsedFromGroup(groupInserted)));
	//						System.out.println("555555 ranking now is (see below):");
	//						System.out.println("555555 TEST SIZE OF CH GROUP - 2 AFTER REMOVAL: "+this.CH.getConceptKeySetFromGroup(groupInserted));
	//						break;
						case "Least Used Model":  
							System.out.println("555555 TEST SIZE OF CH GROUP - 1 BEFORE REMOVAL: "+this.CH.getConceptKeySetFromGroup(groupInserted));
							System.out.println("@@@@@@@@@@");
							System.out.println("33 others for debugging:");
							System.out.println("555555 AGES: "+this.CH.getConceptAgesFromGroup(groupInserted).toString());
							System.out.println("555555 MODIFIED TIMES: "+this.CH.getConceptLastTimeUsedFromGroup(groupInserted).toString());
							System.out.println("@@@@@@@@@@");
							System.out.println("555555 TIMES USED: "+this.CH.getConceptTimesUsedFromGroup(groupInserted).toString());
							System.out.println("555555 EXAMPLES TRAINED: "+this.CH.getConceptExamplesFromGroup(groupInserted).toString());

							// This removes the oldest of the least used classifiers (assuming that min times used per group is shared across many. if not, it removes the least used.)
							ArrayList<Integer> subset = getMinKeys(this.CH.getConceptTimesUsedFromGroup(groupInserted)); // least used classifiers
	//						this.CH.removeLearnerFromGroup(groupInserted, getMinKey(this.CH.getConceptAgesFromGroupSubset(groupInserted, subset))); 
							this.CH.removeLearnerFromGroup(groupInserted, getMinKey(this.CH.getConceptLastTimeUsedFromGroupSubset(groupInserted, subset))); 
							System.out.println("555555 ranking now is (see below):");
							System.out.println("555555 TEST SIZE OF CH GROUP - 2 AFTER REMOVAL: "+this.CH.getConceptKeySetFromGroup(groupInserted));
							break;
						case "FIFO":
							System.out.println("555555 TEST SIZE OF CH GROUP - 1 BEFORE REMOVAL: "+this.CH.getConceptKeySetFromGroup(groupInserted));
	//						System.out.println("555555 AGES: "+this.CH.getConceptAgesFromGroup(groupInserted).toString());
	//						this.CH.removeLearnerFromGroup(groupInserted, getMinKey(this.CH.getConceptAgesFromGroup(groupInserted)));
							System.out.println("555555 AGES: "+this.CH.getConceptLastTimeUsedFromGroup(groupInserted).toString());
							// IMP: As createdOn gives information about the origin of the classifier, there are many cases of recurring ones having sharing creation. 
							// For this reason, FIFO now uses the last time when that classifier was trained as a mechanism to know which one is older (the ID could also be used for this purpose).
							this.CH.removeLearnerFromGroup(groupInserted, getMinKey(this.CH.getConceptLastTimeUsedFromGroup(groupInserted)));
							System.out.println("555555 ranking now is (see below):");
							System.out.println("555555 TEST SIZE OF CH GROUP - 2 AFTER REMOVAL: "+this.CH.getConceptKeySetFromGroup(groupInserted));
							break;
						// TODO: apply theory of cache replacement policies: 
						//	- remove the one trained with less examples (less instancesSeen)
						//  - remove less recently used (LRU). create another variable as createdOn as, reusedOn..
						//  - policy Least frequent recently used (LFRU)
						//  - remove oldest 
						//  - Make them to be distributed over time.
						default:
							break;
					}				
	
				}			
	//			System.out.println("GROUP INSERTED - activeHasLowerErr == "+activeImprovesBestRetrieved+
	//					" on instance: "+this.instancesSeen);
				// If we insert only examples to a single classifier per group, then only crate a classifier if it's the first one in the group
				if (!this.insertExamplesOnlyOption.isSet() | this.CH.getConceptsFromGroup(groupInserted).size() == 0)
					this.CH.addLearnerToGroup(groupInserted, this.active.tmpCopyOfClassifier); // line 28
				else if (this.insertExamplesOnlyOption.isSet()) {
//					System.out.println("concepts from group: "+ this.CH.getConceptsFromGroup(groupInserted).size());
//					System.out.println("assertion "+this.active.instancesTrained+" == "+this.active.conceptBuffer.size() 
//					+" : "+ (this.active.instancesTrained == (long) this.active.conceptBuffer.size()));
					assert this.active.instancesTrained == (long) this.active.conceptBuffer.size();
					System.out.println("Concept buffer size before training: "+this.active.conceptBuffer.size());
					this.CH.trainGroupLearner(groupInserted, this.CH.get(groupInserted).keys().get(0), this.active.conceptBuffer);
					System.out.println("Concept buffer size after training: "+this.active.conceptBuffer.size());

				}
				
			}
	//		else {
	//			System.out.println("GROUP NOT INSERTED - activeHasLowerErr == "+activeImprovesBestRetrieved+
	//					" on instance: "+this.instancesSeen);
	//			System.out.println("GROUP INSERTED : " + groupInserted);
	//			System.out.println("GROUP RETRIEVED : " + groupRetrieved);
	//		}
	//		System.out.println("TEST SIZE OF CH GROUP - 2 AFTER INSERTION: "+this.CH.getConceptsFromGroup(groupInserted).size());
			
			if (this.updateGroupTopologiesOption.isSet()) { // not in v5 of algorithm (line 29 in v4)
				// OJO: esta linea de abajo necesita agregar .clone/.copy en algun sitio, ya que no esta creando objetos nuevos y  hace que el updatetopology de abajo actualice el objeto de grupo de CH
				// this.CH.setGroupTopology(previousTopologyGroupId,
				//		updateTopology(this.CH.copyTopologyFromGroup(previousTopologyGroupId), this.topology.getPrototypes()));
				// asuarez 02-06.2019. quiza esto incluso fuese mas adecuado
				this.CH.setGroupTopology(groupInserted, this.topology.clone()); // pero entonces perderiamos mucho historico..
			}
			
		} else {		
			// Logging each independent case. First lack of prototypes for drift and after, missing examples 
			//  in the warning window and trigger relevant actions according to option values.
			if (this.topology.getNumberOfPrototypesCreated() < this.minTopologySizeForDriftOption.getValue())
				System.out.println("There weren't enough prototypes in the topology.");
		}
	}

	
	// DRIFT ACTIONS

	/***
	 * Register false alarm an update variables consequently
	 * (both the active classifier and the ensemble topology will remain being the same)
	 */
	protected boolean registerDriftFalseAlarm() {
		if (this.eventsLogFile != null && this.logLevel >= 0) logEvent(getFalseAlarmEvent()); // TODO: refactor
		// this.newTopology = null; // then Pn = Pc  (line 30)  // not needed as null by default in drifthandling method
		
		// Change 2 - 31/10/2020: in FA, we train the examples avoided during warning
//		for (int i = 0; i < this.W.size(); i++) {
//			Instance inst = this.W.get(i).copy();
//			if (!this.trainActiveOnWarnOption.isSet())
//				this.active.classifier.trainOnInstance(inst);
//			this.topology.trainOnInstanceImpl(inst);  // TODO: another idea would be to reset the topology here... but then it wouldnt align to the classifier
//		}
		// commented out on 14/01/2021: We should not do this till the warning window finishes and do this.W.delete() 
		//                               or the topology and the active classifier will be trained twice with these examples.
		return true;
	}

	/***
	 * Register recurring drift an update variables consequently
	 * Copy the best recurring learner in the history group passed, and the topology of this group.
	 */
	protected void registerRecurringDrift(Integer indexOfBestRanked, int historyGroup) {
		if (this.eventsLogFile != null && this.logLevel >= 0)
			logEvent(getRecurringDriftEvent(indexOfBestRanked, historyGroup));  // TODO: refactor
		this.active.bkgLearner = this.CH.copyConcept(historyGroup, indexOfBestRanked);
		
		// Change 1 - 31/10/2020: in RC, we train the examples during warning, which belong to the new context
		// if (!this.trainActiveOnWarnOption.isSet())  // shoud this be here?? TODO: read line below
			// TODO otherwise, we train them before and after the drift, these examples will have 
			//  gone to the inserted concept and also are onserted to the retrieved one...
		for (int i = 0; i < this.W.size(); i++) {
			this.active.bkgLearner.classifier.trainOnInstance(this.W.get(i));
		}
		this.newTopology = this.CH.copyTopologyFromGroup(historyGroup);  // then Pn = Ph  (line 30)
		this.newTopology.resetId();
	}

	/***
	 * Register background drift an update variables consequently
	 * Pselected is a new P in case of background drift
	 */
	protected void registerBkgDrift() {
		// Register background drift
		if (this.eventsLogFile != null && this.logLevel >= 0) logEvent(getBkgDriftEvent());  // TODO: refactor
		this.newTopology = this.topology.clone(); // line 30  (new Topology does not create a new object due to the meta class for the clusterer)
		this.newTopology.resetId();
		this.newTopology.resetLearningImpl(); // line 30
	}

	// IDENTIFICATION OF NEXT STATE

	/**
	 * This method ranks all applicable base classifiers in the Concept History (CH)
	 * It also selects the next classifier to be active, or it raises a false alarm
	 * if the drift should be reconsidered.
	 *
	 * It implements the line 30 of the algorithm.
	 *  get topology of the group (methods 'registerDrift...') and retrieve the best classifier
	 *
	 * -----------------------------------------------------------------------------------------------------------------
	 * False alarms depend on the drift decision mechanism
	 * -----------------------------------------------------------------------------------------------------------------
	 *
	 * False alarms are taken into consideration for drifts (the warning will be still active
	 * even if a false alarm is raised for a drift in the same active classifier).
	 * If the background learner is NULL, we consider that the drift signal may have been caused
	 * by a too sensitive drift detection parameterization. In this case, it's clearly
	 * too soon to change the active classifier. Therefore we raise a drift signal.
	 *
	 * We also raise a false alarm when the active classifier obtains less error than the
	 * bkg classifier and all of the classifiers from the CH.
	 *
	 * -----------------------------------------------------------------------------------------------------------------
	 * If the active classifier is not the best available choice / false alarm is
	 * raised, the following logic applies:
	 * -----------------------------------------------------------------------------------------------------------------
	 * If bkgBetterThanCHbaseClassifier == False, the minimum error of the base
	 * classifiers in the CH is not lower than the error of the bkg classifier.
	 * Then, register background drift.
	 *
	 * If CHranking.size() == 0, no applicable concepts for the active classifier in
	 * the concept history. Then, we register background drift. Otherwise, a
	 * recurring drift is the best option.
	 *
	 * @param historyGroup
	 *
	 */
	protected ConceptDrift switchActiveClassifier(ArrayList<Integer> historyGroups) {  // TODO: refactor so false alarm is checked by an specific method
	    int indexOfBestRanked = -1;
		double errorOfBestRanked = -1.0;
		boolean activeImprovesRecurring = false;
		boolean isBkgDrift = false;
		HashMap<Integer, Double> ranking = new HashMap<Integer, Double> ();
		
		// 1 Raise a false alarm for the drift if the background learner is not ready
		if (this.active.bkgLearner == null) {
			System.out.println("False alarm due to lack of BKG classifier on: "+String.valueOf(this.lastDriftOn));  // extra trace by asuarez at 08122019
			return new ConceptDrift(registerDriftFalseAlarm(), false, ranking, indexOfBestRanked, historyGroups.get(0), false);  // first pos is the closest group (we don't compare rankings here)
		}
		// 2 Retrieve best applicable classifier from Concept History (if a CH group applies)
		int historyGroup;
		if (historyGroups.size() > 0 && historyGroups.get(0) != -1) {
			GroupRank bestGroup = rankConceptHistoryClassifiers(historyGroups);
			historyGroup = bestGroup.getGroup();
			ranking = bestGroup.getRank();
		} else historyGroup = -1;
		if (ranking.size() > 0) {
			ArrayList<Integer> indexesOfBestRanked = getMinKeys(ranking); // find indexes of concepts with lowest values (error)
			System.out.println("PRIORITY ARE "+this.priorityOfRecurringClassifiersOption.getChosenLabel()+" CLASSIFIERS.");
			if (this.priorityOfRecurringClassifiersOption.getChosenLabel().equals("Old")) {
				indexOfBestRanked = getMinKey(this.CH.getConceptLastTimeUsedFromGroupSubset(historyGroup, indexesOfBestRanked));
			} else {  // Prioritise the last ones used
				indexOfBestRanked = getMaxKey(this.CH.getConceptLastTimeUsedFromGroupSubset(historyGroup, indexesOfBestRanked));
			}
//			indexOfBestRanked = getMinKey(ranking); // find indexes of concepts with lowest values (error)
			System.out.println("INDEX OF "+this.priorityOfRecurringClassifiersOption.getChosenLabel().toUpperCase() + 
					"EST OF THE BEST CH CLASSIFIERS IS: " + indexOfBestRanked);
			errorOfBestRanked = Collections.min(ranking.values());
			activeImprovesRecurring = activeIsBetterThanCHbaseClassifier(errorOfBestRanked);
		}
		// 3 Compare this against the background classifier and make the decision.
		if (activeIsBetterThanBKGbaseClassifier()) {
			if (ranking.size() > 0 && !activeImprovesRecurring) {
				if(this.multiPassTopologyTrainingOption.isSet()) this.topology.trainUntilFulfillingStoppingCriteria();
				registerRecurringDrift(indexOfBestRanked, historyGroup);
			}
			// False alarm if active classifier is still the best one and when there are no applicable concepts.
			else {
				System.out.println("False alarm, as the active is better than active and the CH of size "+ranking.size()+
						" on  "+String.valueOf(this.lastDriftOn));  // extra trace by asuarez at 08122019
				System.out.println("The err of the best ranked classifier was: "+errorOfBestRanked);
				return new ConceptDrift(registerDriftFalseAlarm(), true, // true as active improves the retrieved
						ranking, indexOfBestRanked, historyGroup, false); 
			}
		} else {
			if(this.multiPassTopologyTrainingOption.isSet()) this.topology.trainUntilFulfillingStoppingCriteria();
			if (ranking.size() > 0 && CHbaseClassifierIsBetterThanBkg(errorOfBestRanked))
				registerRecurringDrift(indexOfBestRanked, historyGroup);
			else {
				registerBkgDrift();
				isBkgDrift = true;
			}
		} 
		return new ConceptDrift(false, activeImprovesRecurring, ranking, indexOfBestRanked, historyGroup, isBkgDrift); // No false alarms raised at this point
	}

	/**
	 * This auxiliary function updates either old or new topologies that will be merged with, compared with, or will replace to the current one.
	 * */
	protected Topology updateTopology(Topology top, Instances w2) {
				
		/* If there is a different stopping criteria, it should depend on the avg error obtained by the topologies
		 * as at some point in the learning process GNG will only improve the accuracy a bit, and slowly, as the approximation should get better overtime.
		 * Stopping early should only give us speed (loosing accuracy). If we do this (we may not need to), the tradeoff is the key.
		int trainType = 0;
		if(stopPercentageOption.getValue() > 0)
			top.stoppingCriteriaOption.setValue((int)((this.stopPercentageOption.getValue() * (double) w2.size()) / 100.0));
		
		if(trainType==0) {
			// We add them several times till achieving the stopping criteria as in iGNGSVM
			// The effect of this would be a GNG topology that may not be able to keep expanding, which may be undesired in NA.
			for (int i=0; top.getNumberOfPrototypesCreated()<top.stoppingCriteriaOption.getValue(); i++){
				top.trainOnInstanceImpl((Instance) w2.get(i));
		        	if(i+1==w2.numInstances()) i = -1;
		    }
		} else { */
			// We add them once
		for (int instPos = 0; instPos < w2.numInstances(); instPos++) {
			top.trainOnInstanceImpl(w2.get(instPos).copy());
			// System.out.println("prototypes created in topology:" + top.getNumberOfPrototypesCreated());
		}
		// }
		return top; // if topology (Pc) is global, then we don't need to return this here
	}
		
    /**
     * This function ranks the best concepts from a given group of the Concept History
     * -----------------------------------
	 * This only takes into consideration Concepts sent to the Concept History
	 * 	after the current classifier raised a warning (see this consideration in reset*)
	 *
	 * The Concept History owns only one learner per historic concept.
	 * 	But each learner has a different window size and error.
	 *
	 * this.indexOriginal - pos of this classifier with active warning in ensemble
	 */
	protected HashMap<Integer, Double> rankConceptHistoryClassifiers(int historyGroup) {
		HashMap<Integer, Double> groupRanking = new HashMap<Integer, Double>();
		for (Concept auxConcept : this.CH.getConceptsFromGroup(historyGroup))
			if (auxConcept.ConceptLearner.internalWindowEvaluator != null
					&& auxConcept.ConceptLearner.internalWindowEvaluator.containsIndex(0)) { // 0 is the pos of the active learner in an ensemble
				groupRanking.put(auxConcept.getHistoryIndex(),
						((DynamicWindowClassificationPerformanceEvaluator) auxConcept.ConceptLearner.internalWindowEvaluator)
								.getFractionIncorrectlyClassified(0));
			}
		System.out.println("XXXXXXX group ranking is: "+groupRanking.toString());
		return groupRanking;
	}
	
	
	protected GroupRank rankConceptHistoryClassifiers(ArrayList<Integer> relevantGroups) {
		
		double minErr = 100.0;
		int bestGroup = -1;
		HashMap<Integer, Double> bestGroupRanking = new HashMap<Integer, Double>();

		for (int i = 0; i < relevantGroups.size(); i++) {
			int historyGroup = relevantGroups.get(i);
			
			HashMap<Integer, Double> groupRanking = new HashMap<Integer, Double>();
			for (Concept auxConcept : this.CH.getConceptsFromGroup(historyGroup)) {
				if (auxConcept.ConceptLearner.internalWindowEvaluator != null
						&& auxConcept.ConceptLearner.internalWindowEvaluator.containsIndex(0)) { // 0 is the pos of the active learner in an ensemble
					double result = ((DynamicWindowClassificationPerformanceEvaluator) 
							auxConcept.ConceptLearner.internalWindowEvaluator).getFractionIncorrectlyClassified(0);
					groupRanking.put(auxConcept.getHistoryIndex(), result);
					System.out.println("Classifier from group "+historyGroup+" has an err of "+result);

					if (result < minErr){
						System.out.println("Classifier from group "+historyGroup+" improves min with an err in the ww of "+result);
						minErr = result;
						bestGroup = historyGroup;
						bestGroupRanking = groupRanking;
					}
				}
			}	
		}
		assert bestGroup != -1;
		System.out.println("The best group is: "+bestGroup);
		System.out.println("XXXXXXX This group's ranking is: "+bestGroupRanking.toString());
		return new GroupRank(bestGroup, bestGroupRanking);
	}
	
	final class GroupRank {
	    private final int group;
	    private final HashMap<Integer, Double> rank;

	    public GroupRank(int group, HashMap<Integer, Double> rank) {
	        this.group = group;
	        this.rank = rank;
	    }

	    public int getGroup() {
	        return group;
	    }

	    public HashMap<Integer, Double> getRank() {
	        return rank;
	    }
	}


	/**
	 * Aux method for getting the best classifier (used to rank classifiers from a group in the CH)
	 * */
	protected Integer getMinKey(Map<Integer, Double> map) {
		Integer minKey = null;
		double minValue = Double.MAX_VALUE;
		for (Integer key : map.keySet()) {
			double value = map.get(key);
			if (value < minValue) {
				minValue = value;
				minKey = key;
			}
		} return minKey;
	}
	
	/**
	 * Aux method for getting a list of the classifiers with the min value of something
	 * */
	protected ArrayList<Integer> getMinKeys(Map<Integer, Double> map) {
		ArrayList<Integer> minKeys = new ArrayList<Integer>();
		double minValue = Double.MAX_VALUE;
		for (Integer key : map.keySet()) {
			double value = map.get(key);
			if (value < minValue) {
				minValue = value;
				minKeys = new ArrayList<Integer>();
				minKeys.add(key);
			} else if (value == minValue) {
				minKeys.add(key);
			}
		} return minKeys;
	}
	
	
	/**
	 * Aux method for getting the worst classifier (used to remove classifiers from a group in the CH)
	 * */
	protected Integer getMaxKey(Map<Integer, Double> map) {
		Integer maxKey = null;
		double maxValue = -Double.MAX_VALUE;
		for (Integer key : map.keySet()) {
			double value = map.get(key);
			if (value > maxValue) {
				maxValue = value;
				maxKey = key;
			}
		} return maxKey;
	}

	protected boolean activeIsBetterThanBKGbaseClassifier() {
		// DEBUG
		System.out.println("[Example " + this.instancesSeen + "] Comparison of classifiers -  WW size: " + this.W.size());
		System.out.println("ACTIVE: "+((DynamicWindowClassificationPerformanceEvaluator) this.active.internalWindowEvaluator)
			.getFractionIncorrectlyClassified(this.active.indexOriginal));
		System.out.println("BKG: "+((DynamicWindowClassificationPerformanceEvaluator)
				this.active.bkgLearner.internalWindowEvaluator).getFractionIncorrectlyClassified(this.active.bkgLearner.indexOriginal));
		System.out.println("activeBetterThanBKGbaseClassifier: "+(((DynamicWindowClassificationPerformanceEvaluator) this.active.internalWindowEvaluator)
			.getFractionIncorrectlyClassified(this.active.indexOriginal) <= ((DynamicWindowClassificationPerformanceEvaluator)
			this.active.bkgLearner.internalWindowEvaluator).getFractionIncorrectlyClassified(this.active.bkgLearner.indexOriginal)));
		
		return (((DynamicWindowClassificationPerformanceEvaluator) this.active.internalWindowEvaluator)
			.getFractionIncorrectlyClassified(this.active.indexOriginal) <= ((DynamicWindowClassificationPerformanceEvaluator)
			this.active.bkgLearner.internalWindowEvaluator).getFractionIncorrectlyClassified(this.active.bkgLearner.indexOriginal));
		// this.indexOriginal - pos of this classifier with active warning in ensemble
		// return ((this.ensemble.evaluator.getFractionIncorrectlyClassified() <= ((DynamicWindowClassificationPerformanceEvaluator)
		//		this.ensemble.bkgLearner.internalWindowEvaluator).getFractionIncorrectlyClassified(this.ensemble.bkgLearner.indexOriginal)));
	}

	protected boolean activeIsBetterThanCHbaseClassifier(double bestFromCH) {
		//DEBUG
		System.out.println("Comparison on instance: "+this.instancesSeen);
		System.out.println("ACTIVE: "+((DynamicWindowClassificationPerformanceEvaluator) this.active.internalWindowEvaluator)
				.getFractionIncorrectlyClassified(this.active.indexOriginal));
			System.out.println("CH: "+bestFromCH);
			System.out.println("activeBetterThanCHbaseClassifier: "+(((DynamicWindowClassificationPerformanceEvaluator) this.active.internalWindowEvaluator)
					.getFractionIncorrectlyClassified(this.active.indexOriginal) <= bestFromCH));
		if(((DynamicWindowClassificationPerformanceEvaluator) this.active.internalWindowEvaluator)
				.getFractionIncorrectlyClassified(this.active.indexOriginal) == bestFromCH){
			System.out.println("IMPORTANT: ACTIVE CLASSIFIER ERR == BEST FROM CH");
		}
		///////////////////// END OF DEBUGGING

		// TODO: RENAME METHOD TO REMOVE THE WORD BETTER, WHICH IS SUBJECTIVE, AND ADD A MORE EXPLICIT NAME WHICH SAYS THAT ERR IS LOWER OR SAME.
		return (((DynamicWindowClassificationPerformanceEvaluator) this.active.internalWindowEvaluator)
			.getFractionIncorrectlyClassified(this.active.indexOriginal) <= bestFromCH);

		// this.indexOriginal - pos of this classifier with active warning in ensemble
		// return (this.ensemble.evaluator.getFractionIncorrectlyClassified() <= bestFromCH); (old comparison)
	}

	protected boolean CHbaseClassifierIsBetterThanBkg(double bestFromCH) {
		// DEBUG
		System.out.println("Comparison on instance: "+this.instancesSeen);
		System.out.println("BKG: "+((DynamicWindowClassificationPerformanceEvaluator) this.active.bkgLearner.internalWindowEvaluator)
				.getFractionIncorrectlyClassified(this.active.bkgLearner.indexOriginal));
		System.out.println("CH: "+bestFromCH);
		System.out.println("bkgBetterThanCHbaseClassifier: "+(bestFromCH <= ((DynamicWindowClassificationPerformanceEvaluator) this.active.bkgLearner.internalWindowEvaluator)
				.getFractionIncorrectlyClassified(this.active.bkgLearner.indexOriginal)));
		
		if (bestFromCH == ((DynamicWindowClassificationPerformanceEvaluator) this.active.bkgLearner.internalWindowEvaluator)
				.getFractionIncorrectlyClassified(this.active.bkgLearner.indexOriginal)) {
			System.out.println("IMPORTANT: BKG CLASSIFIER ERR == BEST FROM CH");
		}
		///////////////////// END OF DEBUGGING
		
		// TODO: RENAME METHOD TO REMOVE THE WORD BETTER, WHICH IS SUBJECTIVE, AND ADD A MORE EXPLICIT NAME WHICH SAYS THAT ERR IS LOWER OR SAME.
		// this.bkgLearner.indexOriginal - pos of bkg classifier if it becomes active in the ensemble (always same pos than the active)
		return (bestFromCH <= ((DynamicWindowClassificationPerformanceEvaluator) this.active.bkgLearner.internalWindowEvaluator)
				.getFractionIncorrectlyClassified(this.active.bkgLearner.indexOriginal));
	}
	
	///////////////////////////////////////
	//
	// LOGGING FUNCTIONS
	// -----------------------------------

	public Event getTrainExampleEvent() {
		String[] eventLog = { String.valueOf(instancesSeen), "Train example", String.valueOf(0),
				String.valueOf(this.active.evaluator.getPerformanceMeasurements()[1].getValue()),
//				this.warningDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), // warning one
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				String.valueOf(this.instancesSeen), String.valueOf(this.active.evaluator.getFractionIncorrectlyClassified()),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings().size(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getNumberOfActiveWarnings(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings(): "N/A"),
				"N/A", 
				"N/A", 
				"N/A",
				"N/A",
				"N/A",
				"N/A"};
		return (new Event(eventLog));
	}
	
	public Event prepareDriftDurationEvent(long duration) {
		System.out.println("#############");
		System.out.println("Duration of Drift in milliseconds: " + duration);
		System.out.println("#############");

		String[] eventLog = { String.valueOf(instancesSeen), "Change of duration --- " +duration , String.valueOf(0),
				String.valueOf(this.active.evaluator.getPerformanceMeasurements()[1].getValue()),
//				this.warningDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), // warning one
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				String.valueOf(this.instancesSeen), String.valueOf(this.active.evaluator.getFractionIncorrectlyClassified()),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings().size(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getNumberOfActiveWarnings(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings(): "N/A"),
				"N/A", 
				"N/A", 
				"N/A",
				"N/A",
				"N/A",
				"N/A"};
		return (new Event(eventLog));
	}

	public Event getWarningEvent() {

		// System.out.println();
		System.out.println("-------------------------------------------------");
		System.out.println("WARNING ON IN MODEL #"+0+". Warning flag status (activeClassifierPos, Flag): "+CH.getWarnings());
		System.out.println("CONCEPT HISTORY STATE AT THE MOMENT OF THIS WARNING  IS: "+CH.classifiersKeySet().toString());
		System.out.println("-------------------------------------------------");
		// System.out.println();

		String[] warningLog = { String.valueOf(this.lastWarningOn), "WARNING-START", // event
				String.valueOf(0),
				String.valueOf(this.active.evaluator.getPerformanceMeasurements()[1].getValue()),
//				this.warningDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), // warning one
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				String.valueOf(this.instancesSeen), String.valueOf(this.active.evaluator.getFractionIncorrectlyClassified()),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings().size(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getNumberOfActiveWarnings(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings(): "N/A"),
				(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.classifiersKeySet().toString(): "N/A"),
				"N/A", 
				"N/A",
				"N/A",
				"N/A",
				"N/A"};
		// 1279,1,WARNING-START,0.74,{F,T,F;F;F;F},...

		return (new Event(warningLog));
	}

	public Event getBkgDriftEvent() {
		System.out.println("DRIFT RESET IN MODEL #"+0+" TO NEW BKG MODEL #"+this.active.bkgLearner.indexOriginal);
		String[] eventLog = {
				String.valueOf(this.lastDriftOn), "DRIFT TO BKG MODEL", String.valueOf(0),
				String.valueOf(this.active.evaluator.getPerformanceMeasurements()[1].getValue()),
//				this.warningDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), // warning one
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				String.valueOf(this.instancesSeen), String.valueOf(this.active.evaluator.getFractionIncorrectlyClassified()),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings().size(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getNumberOfActiveWarnings(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings(): "N/A"),
				"N/A", 
				"N/A", 
				"N/A",
				String.valueOf(this.topology.getInstancesSeenByLearner()),
				String.valueOf(this.topology.getNumberOfPrototypesCreated()),
				"DISABLED DUE TO COMPUTATIONAL COST"}; // String.valueOf(this.topology.getError())};
		return new Event(eventLog);
	}
	
	
	public Event getInsertionToCHEvent(int groupId) {
		System.out.println("INSERTION OF CLASSIFIER'S COPY PRIOR TO WARNING TO GROUP {"+groupId+"} of CH.");
		
		String[] eventLog = {  // TODO: improve the formatting of this log (it's from drift to BKG model + CH.classifiersKeySet)
				String.valueOf(this.lastDriftOn), "INSERTION TO GROUP {"+groupId+"}", String.valueOf(0),
				String.valueOf(this.active.evaluator.getPerformanceMeasurements()[1].getValue()),
//				this.warningDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), // warning one
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				String.valueOf(this.instancesSeen), String.valueOf(this.active.evaluator.getFractionIncorrectlyClassified()),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings().size(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getNumberOfActiveWarnings(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings(): "N/A"),
				(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.classifiersKeySet().toString(): "N/A"),
				"N/A", 
				"N/A",
				"N/A",
				"N/A",
				"N/A"};
		return (new Event(eventLog));
	}
	
	
	public Event getRetrievalFromCHEvent(int groupId) {
		System.out.println("RETRIEVAL OF BEST CLASSIFIER FROM GROUP {"+groupId+"} OF CH TO COMPARE TO ACTIVE AND BKG CLASSIFIERS.");
		
		String[] eventLog = {  // TODO: improve the formatting of this log (it's from drift to BKG model + CH.classifiersKeySet)
				String.valueOf(this.lastDriftOn), "RETRIEVAL FROM GROUP {"+groupId+"}", String.valueOf(0),
				String.valueOf(this.active.evaluator.getPerformanceMeasurements()[1].getValue()),
//				this.warningDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), // warning one
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				String.valueOf(this.instancesSeen), String.valueOf(this.active.evaluator.getFractionIncorrectlyClassified()),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings().size(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getNumberOfActiveWarnings(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings(): "N/A"),
				(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.classifiersKeySet().toString(): "N/A"),
				"N/A", 
				"N/A" ,
				"N/A",
				"N/A",
				"N/A"};
		return (new Event(eventLog));
	}

	public Event getRecurringDriftEvent(Integer indexOfBestRankedInCH, int group) {
		System.out.println(indexOfBestRankedInCH); // TODO: remove after debugging]
		System.out.println("RECURRING DRIFT RESET IN POSITION #"+0+" TO MODEL #"+
				CH.get(group).groupList.get(indexOfBestRankedInCH).historyIndex + " FROM GROUP "+group);
		String[] eventLog = { String.valueOf(this.lastDriftOn), "RECURRING DRIFT", String.valueOf(0),
				String.valueOf(this.active.evaluator.getPerformanceMeasurements()[1].getValue()),
//				this.warningDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), // warning one
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				String.valueOf(this.instancesSeen), String.valueOf(this.active.evaluator.getFractionIncorrectlyClassified()),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings().size(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getNumberOfActiveWarnings(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings(): "N/A"),
				"N/A",
				String.valueOf(this.CH.get(group).groupList.get(indexOfBestRankedInCH).historyIndex),
				String.valueOf(this.CH.get(group).groupList.get(indexOfBestRankedInCH).createdOn),
				String.valueOf(this.topology.getInstancesSeenByLearner()),
			    String.valueOf(this.topology.getNumberOfPrototypesCreated()),
				"DISABLED DUE TO COMPUTATIONAL COST"}; // String.valueOf(this.topology.getError())};
		return (new Event(eventLog));
	}

	public Event getFalseAlarmEvent() {
		/***
		 * This method logs falses alarms, which are classified in three types: 
		 * - Type 1: False alarm on drift. The error of the active classifier is lower or equal
		 * 			  than the best from the CH and bkg in the current window.
		 * - Type 2: Early drift 1: not enough instances received during the warning window. Drift and
		 * 			  warning may have triggered simultaneously. 
		 * - Type 3: Early drift 2: there may be more instances in the warning window than the minimum, but the clustering
		 * 			  algorithm hasn't produced enough prototypes yet. 
		 * 			  (TODO: AN ACTION MAY BE NEEDED IF THIS TYPE APPEARS FREQUENTLY).
		 */
		String msg = "FALSE ALARM ON DRIFT SIGNAL";
		if(this.W.numInstances() < this.minWSizeForDriftOption.getValue())
			msg = "EARLY_DRIFT_1 - NOT ENOUGH EXAMPLES DURING WARNING";
		else if (this.topology.getNumberOfPrototypesCreated() < this.minTopologySizeForDriftOption.getValue())
			msg = "EARLY_DRIFT_2 - NOT ENOUGH PROTOTYPES";
		
		System.out.println("FALSE ALARM IN MODEL #"+0);
		String[] eventLog = { String.valueOf(this.lastDriftOn), msg,
				String.valueOf(0),
				String.valueOf(this.active.evaluator.getPerformanceMeasurements()[1].getValue()),
//				this.warningDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), // warning one
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				String.valueOf(this.instancesSeen), String.valueOf(this.active.evaluator.getFractionIncorrectlyClassified()),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings().size(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getNumberOfActiveWarnings(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings(): "N/A"),
				"N/A", "N/A", "N/A",
				"N/A",
				"N/A",
				"N/A"};
		        // these here added noise to the reports in the case of multipass
//				String.valueOf(this.topology.getInstancesSeenByLearner()),  
//				String.valueOf(this.topology.getNumberOfPrototypesCreated()),
//				String.valueOf(this.topology.getError())};
		return (new Event(eventLog));
	}

	// General auxiliar methods for logging events
	public Event getEventHeaders() {
		String[] headers = { "instance_number", "event_type", "affected_position", // former 'classifier'
				"voting_weight", // voting weight for the three that presents an event.
				"warning_setting", "drift_setting", "affected_classifier_created_on", "error_percentage",
				"amount_of_classifiers", "amount_of_active_warnings", "classifiers_on_warning", "concept_history_snapshot",
				"recurring_drift_to_history_id", "recurring_drift_to_classifier_created_on", "gng_Seen_examples", "gng_prototypes", "topology_error" };
		return (new Event(headers));
	}

	/**
	 * Method to register events such as Warning and Drifts in the event log file.
	 */
	public void logEvent(Event eventDetails) {
		// Log processed instances, warnings and drifts in file of events
		// # instance, event, affected_position, affected_classifier_id last-error, #classifiers;#active_warnings; classifiers_on_warning,
		// 		applicable_concepts_from_here, recurring_drift_to_history_id, drift_to_classifier_created_on
		if (this.eventsLogFile != null) {
			this.eventsLogFile.println(String.join(";", eventDetails.getInstanceNumber(), eventDetails.getEvent(),
					eventDetails.getAffectedPosition(), eventDetails.getVotingWeigth(), // of the affected position
					eventDetails.getWarningSetting(), // WARNING SETTING of the affected position.
					eventDetails.getDriftSetting(), // DRIFT SETTING of the affected position.
					eventDetails.getCreatedOn(), // new, affected_classifier_was_created_on
					eventDetails.getLastError(),
					eventDetails.getNumberOfClassifiers(),
					eventDetails.getNumberOfActiveWarnings(), // #active_warnings
					eventDetails.getClassifiersOnWarning(),  // toString of list of classifiers in warning
					eventDetails.getListOfApplicableConcepts(), // applicable_concepts_from_here
					eventDetails.getRecurringDriftToClassifierID(), // recurring_drift_to_history_id
					eventDetails.getDriftCreation(),
					eventDetails.getTopologyExamples(),
					eventDetails.getTopologyPrototypes(),
					eventDetails.getTopologyError()));
			this.eventsLogFile.flush();
		}
	}

	
	///////////////////////////////////////
	//
	// AUX CLASSES
	// -----------------------------------
	
	/**
	 * Inner class that represents a single tree member of the ensemble. It contains
	 * some analysis information, such as the numberOfDriftsDetected,
	 */
	protected final class NABaseLearner {
		public int indexOriginal;
		public int timesUsed;
		public long createdOn;
		public long instancesTrained;
		public long cumInstTrained; // considering recurrences
		public Classifier classifier;
		ArrayList<Instance> conceptBuffer;
		public boolean isBackgroundLearner;
		public boolean isOldLearner; // only for reference

		// public boolean useBkgLearner; // (now always true in NA)
		// these flags are still necessary at this level
		public boolean useDriftDetector;
		public boolean useRecurringLearner;

		// Bkg learner
		protected NABaseLearner bkgLearner;
		protected double weight;  // weight for voting against it's background learner during warning if the option is enabled

		// Copy of main classifier at the beginning of the warning window for its copy in the Concept History
		protected Concept tmpCopyOfClassifier;

		// Statistics
		public BasicClassificationPerformanceEvaluator evaluator;

		// Internal statistics
		public DynamicWindowClassificationPerformanceEvaluator internalWindowEvaluator; // for bkg and CH classifiers
		protected double lastError;
		protected Window windowProperties;

		private void init(int indexOriginal, Classifier classifier,
				BasicClassificationPerformanceEvaluator evaluatorInstantiated, long instancesSeen, // boolean useBkgLearner,
				boolean useDriftDetector, boolean isBackgroundLearner, boolean useRecurringLearner, boolean isOldLearner,
				Window windowProperties, DynamicWindowClassificationPerformanceEvaluator internalEvaluator, 
				int timesUsed, long cumInstTrained) {

			this.weight = 1.0;
			this.indexOriginal = indexOriginal;
			this.createdOn = instancesSeen;  // creation of bkg learner or creation of active
			this.timesUsed = timesUsed;
			this.cumInstTrained = cumInstTrained;
			this.instancesTrained = 0;

			// only used if insertExamplesOnly is enabled
			if (insertExamplesOnlyOption.isSet()) this.conceptBuffer = new ArrayList<Instance>();

			this.classifier = classifier;
			this.evaluator = evaluatorInstantiated;
			
			this.useDriftDetector = useDriftDetector;
			
			this.isBackgroundLearner = isBackgroundLearner;
			this.useRecurringLearner = useRecurringLearner;
			if (useRecurringLearner) {
				this.windowProperties = windowProperties; // Window params
				this.isOldLearner = isOldLearner; // Recurring drifts
				this.internalWindowEvaluator = internalEvaluator; // only used in bkg and retrieved old classifiers
			}
		}

//		public boolean correctlyClassifies(Instance instance) {
//			return this.classifier.correctlyClassifies(instance);
//		}


		// TODO: have a think on how to propagate 
		public NABaseLearner(int indexOriginal, Classifier classifier, BasicClassificationPerformanceEvaluator evaluatorInstantiated,
				long instancesSeen, boolean useDriftDetector, boolean isBackgroundLearner, boolean useRecurringLearner, boolean isOldLearner,
				Window windowProperties, DynamicWindowClassificationPerformanceEvaluator bkgInternalEvaluator, int timesUsed, long cumInstTrained) {
			init(indexOriginal, classifier, evaluatorInstantiated, instancesSeen, useDriftDetector, isBackgroundLearner,
					useRecurringLearner, isOldLearner, windowProperties, bkgInternalEvaluator, timesUsed, cumInstTrained);
		}

        /**
         *  This function resets the GroCH base classifiers.
		 *
		 *  The steps followed in this method can be seen below:
		 *
		 * - Step 2.1 Update window size in window properties depending on window size inheritance flag (entry parameter/Option)
		 * - Step 2.3 Move copy of active classifier made before warning to Concept History and reset.
		 *			  Its history ID will be the last one in the history (= size)
		 */
		public void reset() {
			System.out.println("-------------------------------------------------");
			System.out.println("RESET (WARNING OFF) IN MODEL #"+this.indexOriginal+
					". Warning flag status (activeClassifierPos, Flag): "+CH.getNumberOfActiveWarnings());
			// System.out.println("-------------------------------------------------");
			this.weight = 1.0;			
			this.instancesTrained = 0;

			if (insertExamplesOnlyOption.isSet()) {
				this.conceptBuffer = new ArrayList<Instance>();
			}
			
			// Transition to the best bkg or retrieved old learner
			if (this.bkgLearner != null) {
				if (this.useRecurringLearner) {
					this.tmpCopyOfClassifier = null; // reset tc.

					// 2.1 Update the internal evaluator properties
					this.bkgLearner.windowProperties.setSize(((this.bkgLearner.windowProperties.rememberWindowSize)
							? this.bkgLearner.internalWindowEvaluator.getWindowSize(this.bkgLearner.indexOriginal)
							: this.bkgLearner.windowProperties.windowDefaultSize));

					// 2.2 Inherit window properties / clear internal evaluator
					this.windowProperties = this.bkgLearner.windowProperties;
				}
				// 2.3 New active classifier is the best retrieved old classifier / clear background learner
				this.classifier = this.bkgLearner.classifier;
				this.evaluator = this.bkgLearner.evaluator;
				this.createdOn = this.bkgLearner.createdOn;
				this.timesUsed = ++this.bkgLearner.timesUsed;
				this.cumInstTrained = this.bkgLearner.cumInstTrained;
				this.bkgLearner = null;
				this.internalWindowEvaluator = null;

			} else {
				this.classifier.resetLearning();
				this.createdOn = instancesSeen;
			}
			this.evaluator.reset();
		}

		public void trainOnInstance(Instance instance) { // Line 5: (x,y) ← next(S)
			this.instancesTrained++;
			this.cumInstTrained++;
			if (insertExamplesOnlyOption.isSet()) this.conceptBuffer.add(instance);
//			System.out.println("Instances trained after drift: "+ this.instancesTrained + "   -  Cumulative for this classifier: " + this.cumInstTrained);
			this.classifier.trainOnInstance(instance); // Line 6: ClassifierTrain(c, x, y) -> Train c on the current										// instance (x, y).
		}
		
		/** Method to be called when inserting to CH when insertExamplesOnlyOption*/
		public void trainOnInstance(Instance instance, boolean insertBuffer) { // Line 5: (x,y) ← next(S)
			this.instancesTrained++;
			this.cumInstTrained++;
			if (insertExamplesOnlyOption.isSet() & insertBuffer) this.conceptBuffer.add(instance);
//			System.out.println("Instances trained after drift: "+ this.instancesTrained + "   -  Cumulative for this classifier: " + this.cumInstTrained);
			this.classifier.trainOnInstance(instance); // Line 6: ClassifierTrain(c, x, y) -> Train c on the current										// instance (x, y).
		}

		public void updateWeight(boolean correctlyClassified, double punishingFactor) { // different to DWM as there is no period 
			if (correctlyClassified) this.weight = this.weight / punishingFactor; // punishingFactor < 0
			else this.weight = this.weight * punishingFactor;
		}
		
		public void resetWeight() {
			this.weight = 1.0;
		}

		public double[] getVotesForInstance(Instance instance) {
//			DoubleVector vote = new DoubleVector(this.classifier.getVotesForInstance(instance));
//			return vote.getArrayRef();
			return this.classifier.getVotesForInstance(instance);
		}
                                                                                                                                  		
		public void createInternalEvaluator() {
			// Add also an internal evaluator (window) in the bkgEvaluator
			this.internalWindowEvaluator = new DynamicWindowClassificationPerformanceEvaluator(
				this.windowProperties.getSize(), this.windowProperties.getIncrements(),
				this.windowProperties.getMinSize(), this.lastError,
				this.windowProperties.getDecisionThreshold(), true, this.windowProperties.getResizingPolicy(),
				this.indexOriginal, "created for active classifier in ensembleIndex #" + this.indexOriginal);
			this.internalWindowEvaluator.setEvaluatorType("ACTIVE");  // for debugging
			this.internalWindowEvaluator.reset();
		}
		
		 /**
		 * This method creates BKG Classifier in warning window
		 * The next steps are followed:
		 *  Step 1 Create a new bkgTree classifier
		 *  Step 2 Resets the evaluator
		 *  Step 3 Create a new bkgLearner object
		 * */
		public void createBkgClassifier(long lastWarningOn) {
		    
			// 1 Create a new bkgTree classifier
			Classifier bkgClassifier = this.classifier.copy();
			bkgClassifier.resetLearning();

			// 2 Resets the evaluator
			BasicClassificationPerformanceEvaluator bkgEvaluator = (BasicClassificationPerformanceEvaluator) this.evaluator.copy();
			bkgEvaluator.reset();
			System.out.println("------------------------------");
			System.out.println("Create estimator for BKG classifier in position: " + this.indexOriginal);
			
			// Add also an internal evaluator (window) in the bkgEvaluator
			DynamicWindowClassificationPerformanceEvaluator bkgInternalWindowEvaluator = null;
			if (this.useRecurringLearner) {
				bkgInternalWindowEvaluator = new DynamicWindowClassificationPerformanceEvaluator(
						this.windowProperties.getSize(), this.windowProperties.getIncrements(),
						this.windowProperties.getMinSize(), this.lastError,  // last error from active classifier also for bkg (for resizing policies > -1)
						this.windowProperties.getDecisionThreshold(), true, this.windowProperties.getResizingPolicy(),
						this.indexOriginal, "created for BKG classifier in ensembleIndex #" + this.indexOriginal);
				bkgInternalWindowEvaluator.setEvaluatorType("BKG");  // for debugging
				bkgInternalWindowEvaluator.reset();
			} System.out.println("------------------------------");

			// 4 Create a new bkgLearner object
			this.bkgLearner = new NABaseLearner(this.indexOriginal, bkgClassifier, 
					bkgEvaluator, lastWarningOn, this.useDriftDetector,
					true, this.useRecurringLearner, false, this.windowProperties, bkgInternalWindowEvaluator, 0, 0);
		}

		/**
		* This method saves a backup of the active classifier that raised a warning
		* to be stored in the Concept History in case of drift.
		*
		* The next steps are followed:
	    	*  Step 1 Update last error before warning of the active classifier. This error is the total fraction of examples
	    	* 		incorrectly classified this classifier was active until now.]
		*  Step 2 Copy Base learner for Concept History in case of Drift and store it temporal object.
		*	 First, the internal evaluator will be null. It doesn't get initialized till once in the Concept History
		*		and the first warning arises. See it in initWarningWindow
		*  Step 3 Add the classifier accumulated error (from the start of the classifier) from the iteration before the warning
		*  A simple concept to be stored in the concept history that doesn't have a running learner.
		*  This doesn't train. It keeps the classifier as it was at the beginning of the training window to be stored in case of drift.
		*/
		public void saveCurrentConcept(long instancesSeen) {
			// step 1
			this.lastError = this.evaluator.getFractionIncorrectlyClassified();
			// step 2
			NABaseLearner tmpConcept = new NABaseLearner(this.indexOriginal, this.classifier.copy(),
					(BasicClassificationPerformanceEvaluator) this.evaluator.copy(), this.createdOn, this.useDriftDetector,
					true, useRecurringLearner, true, this.windowProperties.copy(), null, this.timesUsed, this.cumInstTrained);
			// step 3
			System.out.println("<><><><><>");
			System.out.println("Created TMP copy of active classifier when it had been trained with " + tmpConcept.cumInstTrained + " examples");
			this.tmpCopyOfClassifier = new Concept(tmpConcept, this.createdOn,
					this.evaluator.getPerformanceMeasurements()[0].getValue(), instancesSeen);   // IMP: It saves the acumulated error
			this.tmpCopyOfClassifier.setErrorBeforeWarning(this.lastError);
		}
		
		public void resetConceptBuffer() {
			this.conceptBuffer = new ArrayList<Instance>();			
		}
	}

	protected class Concept {
		protected int ensembleIndex; // position that it had in the ensemble when this supports many active classifiers.
		protected int historyIndex; // id in concept history

		// Stats
		protected long createdOn;
		protected long instancesSeen;
		protected double classifiedInstances;
		protected double errorBeforeWarning;

		// Learner
		public NABaseLearner ConceptLearner;  // TODO: should this be a meta learner?

		// Constructor
		public Concept(NABaseLearner ConceptLearner, long createdOn, double classifiedInstances, long instancesSeen) {
			// Extra info
			this.createdOn = createdOn;
			this.instancesSeen = instancesSeen;
			this.classifiedInstances = classifiedInstances;
			this.ensembleIndex = ConceptLearner.indexOriginal;

			// Learner
			this.ConceptLearner = ConceptLearner;
		}

		public void setHistoryID(int id) {
			this.historyIndex = id;
		}
		
		public void addModelToInternalEvaluator(int ensemblePos, double lastError, int windowSize) {
			this.ConceptLearner.internalWindowEvaluator.addModel(ensemblePos, lastError, windowSize);
		}
		
		public void deleteModelFromInternalEvaluator(int ensemblePos) {
			this.ConceptLearner.internalWindowEvaluator.deleteModel(ensemblePos);
		}
		
		/* The base learner ePos and its window ePW refer to the applicable concept that will be compared from this point to this concept. */
		public void newInternalWindow(NABaseLearner ePos, Window ePW) {
			// Internal windows for CH classifiers during warning. These compare to the active learner (for cases where resizing policy > -1)
			DynamicWindowClassificationPerformanceEvaluator tmpInternalWindow =
					new DynamicWindowClassificationPerformanceEvaluator(ePW.getSize(), ePW.getIncrements(),ePW.getMinSize(),
							ePos.lastError, ePW.getDecisionThreshold(), ePW.getDynamicWindowInOldClassifiersFlag(), ePW.getResizingPolicy(),
							ePos.indexOriginal, "created for old-retrieved classifier in ensembleIndex #" + ePos.indexOriginal);
			tmpInternalWindow.setEvaluatorType("CH");
			tmpInternalWindow.reset();
			this.setInternalEvaluator(tmpInternalWindow);
		}
		
		public int getAmountOfApplicableModels() {
			return this.ConceptLearner.internalWindowEvaluator.getAmountOfApplicableModels();
		}
		
		public void addResultToInternalEvaluator(InstanceExample inst, double[] voteRef) {
			this.ConceptLearner.internalWindowEvaluator.addResult(inst, voteRef);
		}

		public NABaseLearner getBaseLearner() {
			return this.ConceptLearner;
		}
		
		// Getters
		
		public int getEnsembleIndex() {
			return this.ensembleIndex;
		}

		public int getHistoryIndex() {
			return this.historyIndex;
		}
		
		public DynamicWindowClassificationPerformanceEvaluator getInternalEvaluator() {
			return this.ConceptLearner.internalWindowEvaluator;
		}
		
		public long getCreatedOn() {
			return this.createdOn;
		}

		
		public long getTimesUsed() {
			return this.ConceptLearner.timesUsed;
		}
		
		public long getNumberOfTrainingExamples() {
			return this.ConceptLearner.cumInstTrained;
		}

		// Setters
		public void setErrorBeforeWarning(double value) {
			this.errorBeforeWarning = value;
		}
		
		public void setInternalEvaluator(DynamicWindowClassificationPerformanceEvaluator intEval) {
			this.ConceptLearner.internalWindowEvaluator = intEval;
		}
		
	}
	
	/** 
	 * A topology represents an state of the data stream. 
	 * The state is represented by a set of prototypes (or cluster centers) which, all together (not each center), 
	 *  define a group of classifiers (a group of concepts). This is, there is an n to 1 relationship between prototypes (centers) to each group. 
	 *  And each group contains n classifiers.
	 * */
	protected class Topology implements Cloneable {
		
		protected GNG learner; // TODO: this should be a meta class of cluster instead
		protected Instances buffer; // this buffer works when the learner is disabled (if using all examples instead of topologies)
		private Instance auxInst;
		private int id;
		private Integer [] attributeSubset;
		private boolean useAttributeSubset;
		private boolean trainsOnline;		
		private ArrayList<Instance> offlineTrainingBuffer;
		Instances subsetHeaders;
		private boolean useMahalanobisObj;
		private Attribute [] subsetAttributes;

		public Topology(ClassOption topologyLearnerOption, Integer[] topologyAttributeSubset, MahalanobisDistanceSingleMatrix m_dist, boolean trainsOnline) {
			this.learner = (GNG) getPreparedClassOption(topologyLearnerOption);  // TODO: make it a generic clustering algorithm (not casted as GNG)
			this.id = (int) (Math.random() * 1000000);
			this.trainsOnline = trainsOnline;
			if (topologyAttributeSubset[0] != -1) {
				this.attributeSubset = topologyAttributeSubset;
				this.useAttributeSubset = true;
			} else this.useAttributeSubset = false;
			this.useMahalanobisObj = true;
			this.learner.setMahalanobisDistObj(m_dist); // the underlying learner needs to implement this method
		}
		
		public void trainUntilFulfillingStoppingCriteria() {
			// Train buffer first?
			if (!this.trainsOnline) trainFromBuffer();
			if (this.learner != null) {
				this.learner.trainUntilFulfilled(); // method only implemented in GNG. this needs to be changed to use another learner.
			}			
		}

		public Topology(ClassOption topologyLearnerOption, Integer[] topologyAttributeSubset, boolean trainsOnline) {
			this.learner = (GNG) getPreparedClassOption(topologyLearnerOption);  // TODO: make it a generic clustering algorithm (not casted as GNG)
			this.id = (int) (Math.random() * 1000000);
			this.trainsOnline = trainsOnline;
			if (topologyAttributeSubset[0] != -1) {
				this.attributeSubset = topologyAttributeSubset;
				this.useAttributeSubset = true;
			} else this.useAttributeSubset = false;
		}
		
		public Topology(Integer[] topologyAttributeSubset, boolean trainsOnline) {
			System.out.println("Creating topology container without learner.");
			this.learner = null;
			this.trainsOnline = trainsOnline;
			this.id = (int) (Math.random() * 1000000);
			if (topologyAttributeSubset[0] != -1) {
				this.attributeSubset = topologyAttributeSubset;
				this.useAttributeSubset = true;
			} else this.useAttributeSubset = false;		}

		protected void resetId() {
			this.id = (int) (Math.random() * 1000000);
		}
		
		protected void resetLearningImpl() {
			if(!this.trainsOnline) this.offlineTrainingBuffer = new ArrayList<Instance>();
			if (this.learner != null) this.learner.resetLearningImpl();
			else if (this.buffer != null && this.buffer.numInstances() > 0) this.buffer.delete();
		}
		
		public void trainOnInstanceImpl(Instance inst) {
			Instance trainingInstance = inst.copy();
			if (this.useAttributeSubset) trainingInstance = applyAttributeSubset(trainingInstance);
			if (this.auxInst == null) {
				this.auxInst = trainingInstance.copy(); // Aux variable for conversions
				if (this.useMahalanobisObj) this.learner.setAuxInstance(trainingInstance.copy());
			}
			
			if (!this.trainsOnline) {
				if (this.offlineTrainingBuffer == null) this.offlineTrainingBuffer = new ArrayList<Instance>();//new Instances ((trainingInstance.copy()).dataset());
				this.offlineTrainingBuffer.add(trainingInstance);
//				System.out.println("add instance to buff size: "+this.offlineTrainingBuffer.size());
			} else {
				if (this.learner != null) {
					this.learner.trainOnInstanceImpl(trainingInstance);
				} else {
					// DEPRECATED BUFFER
					assert false; // raise an exception if the code reaches here as this is a deprecated feature
					if (this.buffer == null) this.buffer = new Instances ((trainingInstance.copy()).dataset());
					this.buffer.add(trainingInstance);  
					// TODO: 04-02-2020: The row above delays the whole algorithm. 
					//		 It may need to be improved performance wise (converting it to instances/matrix at a later stage).
				}
			}
		}
		
		public void trainFromBuffer() {
			System.out.println("THE BUFFER HAS " + this.offlineTrainingBuffer.size() + " INSTANCES.");
			for (int i = 0; i < this.offlineTrainingBuffer.size(); i++) {
//				if (i%100 == 0) System.out.println(i);
				this.learner.trainOnInstanceImpl(this.offlineTrainingBuffer.get(i));
			} System.out.println("Trained with " + this.offlineTrainingBuffer.size() + " examples");
			this.offlineTrainingBuffer = new ArrayList<Instance>();
			System.out.println("Now there are " + this.offlineTrainingBuffer.size() + " examples");
			System.out.println("===========");
			System.out.println("==888888888888=");
			System.out.println("===========");
			System.out.println("===========");
		}
	
		protected int getNumberOfPrototypesCreated() {
			if (this.learner != null) return  ((GNG) this.learner).getNumberOfPrototypesCreated();
			else if (this.buffer == null) return 0;
			else {
				System.out.println("Warning: As the topology learner is disabled, "
						+ "the number returned corresponds to examples rather than prototypes");
				return this.buffer.numInstances();
			}
		}
		
		protected Instances prototypesToUnsupervisedInstances(ArrayList<GUnit> tmpPrototypes) {
			Instances tmp = new Instances ((this.auxInst.copy()).dataset());
			tmp.delete();
			Instance inst = null;

			for(GUnit prototype: tmpPrototypes) {
				inst = (Instance) this.auxInst.copy();
//				System.out.println(inst.numAttributes());
//				System.out.println(prototype.w.length);
				for (int j = 0; j < prototype.w.length; j++) {
//					System.out.println("j: "+j+" inst_val:"+inst.value(j)+" prot_val:"+prototype.w[j]);
					inst.setValue(j, prototype.w[j]);
//					System.out.println("j: "+j+" inst_val:"+inst.value(j));
				}
				tmp.add(inst);
			}
			return tmp;
		}
		
		public Instances getPrototypes() {
			if (this.learner != null) return prototypesToUnsupervisedInstances(this.learner.getS()); 
			else return getBuffer();
		}
		
		public Instances getBuffer() {
			return this.buffer;
		}
		
		public void setLearner (GNG newLearner) {
			this.learner = (GNG) newLearner;
		}
		
		public Attribute [] getSubsetAttributes() {
			return this.subsetAttributes;
		}
		
		public GNG getLearner () {
			return this.learner;
		}
		
		public double getError() {
			if (this.learner != null) return this.learner.getQuantizationError();
			else return -1.0;
		}
		
		public long getInstancesSeenByLearner() {
			if (this.learner != null) return this.learner.getInstancesSeen();
			else return this.buffer.numInstances();
		}
		
		public int getLearnerId() {
			if (this.learner != null) return this.learner.id;
			else return -1;
		}
		
		public void setBuffer(Instances buff) {
			this.buffer = new Instances (buff.get(0).copy().dataset());
			for(int i = 0; i < buff.numInstances(); i++) {
				this.buffer.add(buff.get(i));
			} 
		}
		
		@Override
		protected Topology clone() {
		    try {
	    		Topology cloned = (Topology) super.clone();
	    		if (cloned.getLearner() != null) 
	    			cloned.setLearner(cloned.getLearner().clone());
	    		else if(cloned.getBuffer() != null && cloned.getNumberOfPrototypesCreated() > 0) 
	    			cloned.setBuffer(cloned.getBuffer());
	    		return cloned;
			} catch (CloneNotSupportedException e) {
				e.printStackTrace();
			} return null;
		}
		
		/*
		 * This method reduces the feature set for a given instance.
		 * */
	  	 @SuppressWarnings("null")
		 private Instance applyAttributeSubset(Instance inst) { 
		    // 1. Iterate through attributes according to the subset and modify instance
			double classValueAux = inst.classValue();
			Instances subSetHeadersAux = new Instances(inst.dataset());
			int [] v_pos = new int [(this.attributeSubset.length + 1)];
			Attribute [] v_att = new Attribute [(this.attributeSubset.length + 1)]; 
			double [] v_val = new double [(this.attributeSubset.length+ 1)];
			for (int j = 0; j <= this.attributeSubset.length; j++) {
				v_pos[j] = j;
				if (j < this.attributeSubset.length) {
					if(this.attributeSubset[j] == inst.classIndex()) {
						System.out.println("ISSUE FOUND! Label included as part of the feature subset, "
								+ "and therefore this attribute is duplicated!");
						v_val[j] = (Double) null;
						v_att[j] = null;
					} else if (this.attributeSubset[j] > inst.classIndex())
						System.out.println("ISSUE FOUND! Out of range in feature subset. "
								+ "At least one of the specified indexed attributes point to an  attribute that does not exist!");
					else {
						v_val[j] = inst.value(this.attributeSubset[j]);  // we may need to change this to names..
						v_att[j] = inst.attribute(this.attributeSubset[j]);	
					}
				} else {
					v_val[j] = inst.classValue();
					v_att[j] = inst.classAttribute();
				}
			} inst = new InstanceImpl(1, v_val, v_pos, v_pos.length); // @author: asuarez: parece que esta es la forma de hacerlo...
			
			// 2. Modify headers of dataset (TODO: do this only once and send the dataset each time an object is created, or/and to the CH once) 
			if (this.subsetHeaders == null) {
				this.subsetHeaders = subSetHeadersAux;
				this.subsetHeaders.setAttributes(v_att);  
				this.subsetHeaders.setClassIndex(v_att.length - 1); 
				this.subsetAttributes = v_att;
			}
			inst.setDataset(subsetHeaders);  // add this. to topologyDataset
			inst.setClassValue(inst.numAttributes() - 1, classValueAux);  
			return inst;
		}
	}

	
	protected class Group {
		int id;
		Topology conceptCluster;
		public ConcurrentHashMap<Integer, Concept> groupList; // List of concepts per group

		public Group(int id, Topology top, Integer [] attributeSubset) {
			this.id = id; // nextID();
			this.conceptCluster = top;
			this.groupList = new ConcurrentHashMap<Integer, Concept>();
		}

// 		// this wouldn't need to be refactored for cases when the concept learner of the topology is disabled
//		public Group(int id, ClassOption topologyLearnerOption, Integer [] attributeSubset ) {
//			this.id = id; // nextID();
//			this.conceptCluster = new Topology(topologyLearnerOption, attributeSubset);
//			this.groupList = new ConcurrentHashMap<Integer, Concept>();
//		}
		
		public void init(Instances tmpPrototypes) {
			for (int instPos = 0; instPos < tmpPrototypes.size(); instPos++) {
				this.conceptCluster.trainOnInstanceImpl(tmpPrototypes.get(instPos));
			}
		}

		public NABaseLearner copyConcept(int key) {
			NABaseLearner aux = this.groupList.get(key).getBaseLearner();
			return aux;
		}

		// Getters
		public int getID() {
			return this.id;
		}

		public Instances getTopologyPrototypes() {
			return this.conceptCluster.getPrototypes(); // these instances don't belong to a class (unsupervised)
		}
		
		public void setConceptCluster(Topology topology2) {
			this.conceptCluster = topology2;
		}
		
		public Topology getConceptCluster() {
			return this.conceptCluster;
		}
		
		public Concept getConcept(int id) {
			return groupList.get(id);
		}
		
		public Collection<Concept> values(){
			return this.groupList.values();
		}
		
		public void put(int idx, Concept learner) {
			this.groupList.put(idx, learner);
		}
		
		public void remove(int learnerID) {
			this.groupList.remove(learnerID);
		}
		
		public ArrayList<Integer> keys() {
			return Collections.list(this.groupList.keys());
		}
	
	}

	/***
	 * Static and concurrent for all DTs that run in parallel Concept_history = (list of concept_representations)
	 */
	protected class ConceptHistory {
		
		boolean useTopologyAttributeSubset;
		Integer [] topologyAttributeSubset; 
		double maxDistanceThreshold;
		String distanceMetric;
		boolean computeSingleMatrix;
		MahalanobisDistanceSingleMatrix m_dist;  // only useful here if computeSingleMatrix = True
		protected boolean debugNN = true;

		// Concurrent Concept History List
		protected HashMap<Integer, Group> history; // now this is a list of groups

		// List of ensembles with an active warning used as to determine if the history list evaluators should be in use
		protected HashMap<Integer, Boolean> classifiersOnWarning;
		
//		public ConceptHistory() {
//			this.history = new HashMap<Integer, Group>();
//			this.classifiersOnWarning = new HashMap<Integer, Boolean>();
//			this.maxDistanceThreshold = Float.MAX_VALUE;
//		}
		
		public ConceptHistory(double distThreshold, Integer [] topologyAttributeSubset, 
				String distanceMetric, String matrixModality, String [] preTrainDatasetPaths) {
			this.history = new HashMap<Integer, Group>();
			this.classifiersOnWarning = new HashMap<Integer, Boolean>();
			this.maxDistanceThreshold = distThreshold;
			this.topologyAttributeSubset = topologyAttributeSubset;
			this.distanceMetric = distanceMetric;
			this.computeSingleMatrix = matrixModality.equals("1 matrix only");
			if (this.topologyAttributeSubset.length > 1 && this.topologyAttributeSubset[0] != -1) {
				this.useTopologyAttributeSubset = true;
				if (this.computeSingleMatrix) this.m_dist = new MahalanobisDistanceSingleMatrix(preTrainDatasetPaths, this.topologyAttributeSubset);
			} else {
				if (this.computeSingleMatrix) this.m_dist = new MahalanobisDistanceSingleMatrix(preTrainDatasetPaths);
			}
		}
		

		public void printTopologies() {
			for (Group g : this.history.values()) {
				System.out.println("*******************");
				System.out.println("Learner ID: "+g.getConceptCluster().getLearnerId());
				System.out.println("TOP ID: "+g.getConceptCluster().id);
				System.out.println(g.getTopologyPrototypes().numInstances()+" prototypes in group "+g.getID());
				System.out.println("*******************");
			}
		}
		
		public NABaseLearner copyConcept(int group, int key) {
			NABaseLearner aux = history.get(group).copyConcept(key);
			return aux;
		}
		
		public int getNumberOfActiveWarnings() {
			int count = 0;
			for (Boolean value : classifiersOnWarning.values()) if (value) count++;
			return count;
		}
		
		/**
		 * When the concept is added the first time, it doesn't have applicable classifiers.
		 * They are not inserted until the first warning. So the Concept History only runs over warning windows.
		 */
		public void updateHistoryErrors(Instance inst) {
			for (int historyGroup : this.history.keySet()) {
				for (Concept learner : this.history.get(historyGroup).groupList.values()) {
					DoubleVector oldLearnerVote = new DoubleVector(learner.ConceptLearner.getVotesForInstance(inst));
					if (learner.getInternalEvaluator() != null && learner.getInternalEvaluator().getAmountOfApplicableModels() > 0)
						learner.addResultToInternalEvaluator(new InstanceExample(inst), oldLearnerVote.getArrayRef());
				}
			}
		}
		
		
		/**
		 * This updates the error of CH classifiers with warning.
		 * The next steps are followed:
		 * - 1 It turns on windows flag in the Concept History.
		 * 		Also, if the Concept History is ready and it contains old classifiers,
		 * 		it adds both prior estimation and window size to each concept history learner.
		 * - 2 If the concept internal evaluator has been initialized for any other classifier on warning,
		 * 		add window size and last error of current classifier on warning.
		 * - 3 Otherwise, initialize a new internal evaluator for the concept
		 *
		 * @parameter ePos: base classifier on a given ensemble position
		 * @parameter lastError of the above-mentioned base classifier
		 */
		public void increaseNumberOfWarnings(int ensemblePos, NABaseLearner ePos, double lastError) {
			this.classifiersOnWarning.put(ePos.indexOriginal, true);
			
			Window ePW = ePos.windowProperties;
			if (history != null && history.size() > 0) {
				// This adds it as an applicable concept to all the groups, as we don't know its group yet
				for (int historyGroup : history.keySet()) { // TODO: see in method decreaseNumberOfWarnings
					for (Concept learner : history.get(historyGroup).groupList.values()) {
						if (learner.getInternalEvaluator() != null) { // Step 2
							System.out.println("ADDING VALUES TO INTERNAL EVALUATOR OF CONCEPT "+learner.historyIndex+" IN POS "+ePos.indexOriginal);
							learner.addModelToInternalEvaluator(ensemblePos, lastError, ePW.windowSize);

						} else { // Step 3: Otherwise, initialize a new internal evaluator for the concept
//							System.out.println("INSTANCIATING FOR THE FIRST TIME INTERNAL EVALUATOR FOR CONCEPT "+learner.historyIndex+" IN POS "+ePos.indexOriginal);
							learner.newInternalWindow(ePos, ePW);
						}
					}
				}
			}
		}
		
		/**
		 * This method decreases the amount of warnings in concept history and from evaluators
		 * */
		public void decreaseNumberOfWarnings(int ensemblePos) {
			this.classifiersOnWarning.put(ensemblePos, false);
			if (this.history != null && this.history.size() > 0) {
				// This adds it as an applicable concept to all the groups, as we don't know its group yet (the term 'applicable concepts' is only relevant if there is more than 1 base classfiers.)
				// TODO: performance improvement: can we have only internal evaluators between classifiers inside a given group and not across groups?
				for (int historyGroup : history.keySet()) {
					for (Concept learner : this.history.get(historyGroup).values()) {
						// TODO: performance improvement: can we use use oldGroup instead of historyGroup? (then change it also when increasing warnings)
						if (learner.getInternalEvaluator() != null && learner.getInternalEvaluator().containsIndex(ensemblePos)) {
							learner.deleteModelFromInternalEvaluator(ensemblePos);
							if (learner.getAmountOfApplicableModels() == 0) learner.setInternalEvaluator(null);  // TODO: implement this in RCARF & Evolving RCARF once tested
						}
					}
				}
			}
		}
		
		/** Creation of new group and pushing of this to the CH
		protected void createNewGroup(int groupId, Instances tmpPrototypes, ClassOption learnerOption) { // , Topology newTopology) {
			Group g = new Group(groupId, learnerOption);
			g.init(tmpPrototypes);
			this.history.put(groupId, g); // the id is there twice to keep track of it and for testing purposes.
		}*/
		protected void createNewGroup(int groupId,  Topology top, Integer [] topSubset) {
//			System.out.println("CREATING NEW TOPOLOGY");
			Group g = new Group(groupId, top, topSubset);
			
//			System.out.println();
//			System.out.println("SAVING TOPOLOGY:" + g.conceptCluster.getNumberOfPrototypesCreated());
			
			this.history.put(groupId, g); // the id is there twice to keep track of it and for testing purposes.
		}
		
		public boolean hasGroups() {
			return (this.history.size() > 0);
		}
		
		/**
		 * This method receives the current list of training examples received during
		 * the warning window and checks what's the closest group.
		 */
		protected ArrayList<Integer> findGroups(Instances w, boolean usingPrototypesFlag, boolean mc) {
			double min = Double.MAX_VALUE;
			double dist = 0;
//			int group = -1;
//			ArrayList<Integer> groups = new ArrayList<Integer>();
			HashMap<Double, Integer> groupsDistance = new HashMap<Double, Integer>();
			Instances w1 = new Instances(w);
			
			if (this.useTopologyAttributeSubset && !usingPrototypesFlag) {
//				System.out.println(w1.get(0).classAttribute().name());
//				System.out.println(w1.get(0).classIndex());
//				System.out.println(w1.get(0).classValue());
//				System.out.println(w1.classIndex());
				for (int i = 0; i < w1.numInstances(); i++) 
					w1.set(i, this.history.get(0).getConceptCluster().applyAttributeSubset(w1.get(i).copy()));
				w1.setAttributes(this.history.get(0).getConceptCluster().getSubsetAttributes());
				w1.setClassIndex(w1.get(0).numAttributes() - 1);
//				System.out.println(w1.get(0).classAttribute().name());
//				System.out.println(w1.get(0).classIndex());
//				System.out.println(w1.get(0).classValue());
//				System.out.println(w1.classIndex());
			}
			
			// asuarez: Debugging groups
			System.out.println();
			System.out.println("There were " + w1.numInstances() + " examples or prototypes for comparison to the CH topologies.");
			System.out.println("There are " + this.history.size() + " groups.");
			
			for (Group g : this.history.values()) {
				System.out.println("+++++++");
				// Mahalanobis will consider classes as any other attribute for the distances 
				// Euclidean will in computeDistances(), but it may not in LinearNNSearch. TODO: check and fix if required
				if (this.distanceMetric.equals("Euclidean")) {
					dist = getMeanDistanceToNN(w1, g.getTopologyPrototypes()); 
				} else if (this.distanceMetric.equals("Mahalanobis")) {
					if (this.computeSingleMatrix) {
						dist = this.m_dist.distance(w1, g.getTopologyPrototypes());
					} else {
						SamoaToWekaInstanceConverter converter = new SamoaToWekaInstanceConverter();
						MahalanobisDistance m = new MahalanobisDistance();  // TODO: move this up
						dist = m.distance(converter.wekaInstances(w1), 
									      converter.wekaInstances(g.getTopologyPrototypes()));
					} // Debugging
					System.out.printf("Mahalanobis distance between matrices A nd B is %.3f", dist);
					System.out.println();

				} else {
					System.out.println("Please, specify a distance metric for concept similarity.");
				}
				
				System.out.println("insert? " + (dist < this.maxDistanceThreshold));
				System.out.println("min? " + (dist < min) );
				System.out.println("MC? " + mc);

				if ((dist < this.maxDistanceThreshold) & ((dist < min) || mc)) {
					// groupsDistance should not be large in size. Thus, it should not be required to create a new arraylist each time.
					if (!mc) groupsDistance.clear(); // for insertions, and for retrievals if multi-cluster option is not enabled, only allow 1 group at a time.
					min = dist;
					groupsDistance.put(dist, g.getID());
					System.out.println("Adding group: "+g.getID());
				}
				System.out.println("new size grouphashmap: "+groupsDistance.size());
				System.out.println("Distance of instances in W to group "+g.getID()+" is "+dist);
				System.out.println("Learner ID: "+g.getConceptCluster().getLearnerId());
				System.out.println(g.getTopologyPrototypes().numInstances()+" prototypes in group "+g.getID());
				System.out.println("Number of historical classifiers in the group: "+g.groupList.size());
				System.out.println("++++++");
			} 
			ArrayList<Integer> groups = null;
			if (groupsDistance.size() <= 1)  groups = new ArrayList<>(groupsDistance.values());
			else groups = new ArrayList<>(new TreeMap<Double, Integer>(groupsDistance).values());
			
			System.out.println("all info is: "+groupsDistance.toString());
			System.out.println("The groups inside the dist thresh (in asc order by distance) are: " + groups.toString());
			if (groups.size() > 0) assert groupsDistance.get(min) == groups.get(0);  // make sure that the output is ordered by dist
			
			// Add a pos with -1 if empty, to signal a failed retrieval (so the legacy code does not break)
			if (groups.size() == 0) groups.add(-1);
			assert mc | groups.size() == 1;  // if mc enabled, there will be 1-n groups. otherwise just one (-1 if none)
			assert ((groups.get(0) == -1) && (groups.size() == 1)) | (groups.get(0) >= 0); // if no groups found in the interval, the size of the array should be 1.
			
			if (groups.get(0) == -1) System.out.println("No groups selected as the min distance is "+min+" and the maximum dist allowed is: "+this.maxDistanceThreshold);
			else System.out.println("The selected groups before Concept Similarity (comparing accuracy for the warning window) are: "+groups.toString());
			if (min < this.maxDistanceThreshold) {
				System.out.println("The selected groups before Concept Similarity (comparing accuracy for the warning window) are: "+groups.toString());
			} else {
				System.out.println("No groups selected as the min distance is "+min+
						" and the maximum dist allowed is: "+this.maxDistanceThreshold);				
			} return groups;
		}
		
		/**
		 * This function computes the sum of the distances between every prototype in
		 * the topology passed and the current list of instances during warning (W)
		 */
		/*protected double getMeanDistance(Instances w1, Instances w2) {
			double [] dist = new double[w1.numInstances()];
			double totalDist = 0.0;
			for (int instPos1 = 0; instPos1 < w1.numInstances(); instPos1++) {
				for (int instPos2 = 0; instPos2 < w2.numInstances(); instPos2++) {
					dist[instPos1] += computeDistance(w1.get(instPos1), w2.get(instPos2));
				} dist[instPos1] = dist[instPos1] / w2.numInstances();
			} 
			// Averaging distances
			for (int i = 0; i < dist.length; i++)  totalDist += dist[i];
			return totalDist / w1.numInstances();
		}*/
		
		/**
		 * This function computes the sum of the distances between the nearest prototype in a group's topology 
 		 *  and the current list of instances during warning (W)
		 */
		@SuppressWarnings("null")
		protected double getMeanDistanceToNN(Instances w1, Instances topologyPrototypes) {
			int nPrototypes = 1; // number of neighbors
			Instances nearestPrototypes;
			try {
				EuclideanDistanceModified d = new EuclideanDistanceModified(w1);
				double [] dist = new double[w1.numInstances()];
				double totalDist = 0.0;
				for (int i = 0; i < w1.numInstances(); i++) {
					nearestPrototypes = getNearestPrototypes(w1.get(i), topologyPrototypes, nPrototypes);
					for (int j = 0; j < nearestPrototypes.numInstances(); j++) {
						dist[i] += d.distanceUsingClass(w1.get(i), nearestPrototypes.get(j)); // squared distance (default in GNG)
					} dist[i] = dist[i] / nearestPrototypes.numInstances(); // divided by 1 if only 1 neighbor (default)
					if (this.debugNN) assert nearestPrototypes.numInstances() == nPrototypes; 
				} 
				// Averaging distances
				for (int i = 0; i < dist.length; i++)  totalDist += dist[i];
				return totalDist / w1.numInstances();
				
			} catch (Exception e) {
				e.printStackTrace();
			}  
			return (Double) null; 
		}

		public Instances getNearestPrototypes(Instance i, Instances topologyPrototypes, int nPrototypes) throws Exception {		
			// Instantiate NN search object and return results 
			LinearNNSearchModified m_NNSearch = new LinearNNSearchModified();  // object modified to take instances into account
		    m_NNSearch.setInstances(topologyPrototypes);
		    //TODO: distances using a kernel => m_NNSearch.getDistances();
		    return m_NNSearch.kNearestNeighboursUsingClass(i, nPrototypes);
		}

		public int size() {
			return this.history.size();
		}
		
		
		/**
		 * This returns a HashMap of keySets per groups created
		 * */	
		public HashMap<Integer, Set<Integer>> classifiersKeySet(){
			HashMap<Integer, Set<Integer>> keySets = new HashMap<Integer, Set<Integer>>();
			for (Integer key: this.history.keySet()) {
				keySets.put(key, this.history.get(key).groupList.keySet());
				// for debugging
//				System.out.println("#########################");
//				System.out.println(this.history.get(key).groupList.keys().toString());
//				System.out.println("##");
//				if (this.history.get(key).groupList.size() > 0) System.out.println(this.history.get(key).groupList.get(0).historyIndex);
//				if (this.history.get(key).groupList.size() > 0) System.out.println(this.history.get(key).groupList.get(0).ensembleIndex);
//				System.out.println("------------------------");
//				System.out.println(this.history.get(key).getConceptCluster().id);
//				System.out.println();
			}	
			return keySets;
		}
		
		private Group get(int key) {
			return this.history.get(key);
		}
		
		public HashMap<Integer, Boolean> getWarnings() {
			return this.classifiersOnWarning;
		}
		
		public Topology copyTopologyFromGroup(int key) {
			return this.get(key).getConceptCluster().clone();
		}
				
		public void setGroupTopology(int groupID, Topology top) {
			this.history.get(groupID).setConceptCluster(top); // line 26
		}
		
		public void addLearnerToGroup(int groupID, Concept learner) {
			this.history.get(groupID).put(learner.historyIndex, learner); // line 28
		}
		
		/** This method trains a set of examples over a classifier in the concept history*/
		public void trainGroupLearner(int groupID, int classifierID, ArrayList<Instance> conceptBuffer) {
			System.out.println(":::::::::::::::::::::::::::");
			System.out.println("inst Trained before buffer:" + this.history.get(groupID).getConcept(classifierID).getBaseLearner().instancesTrained);
			Concept cnpt = this.history.get(groupID).getConcept(classifierID);
			for (int i = 0; i < conceptBuffer.size(); i++) {
				cnpt.getBaseLearner().trainOnInstance(conceptBuffer.get(i), false);
			} 
			System.out.println("Internal concept buffer CH was: "+cnpt.getBaseLearner().conceptBuffer.size());
			cnpt.getBaseLearner().resetConceptBuffer();
			System.out.println("Internal concept buffer CH now is: "+cnpt.getBaseLearner().conceptBuffer.size());
			cnpt = null;
			conceptBuffer = new ArrayList<Instance>();
			System.out.println("inst Trained after buffer:" + this.history.get(groupID).getConcept(classifierID).getBaseLearner().instancesTrained);
			System.out.println(":::::::::::::::::::::::::::");
		}
		
		public void removeLearnerFromGroup(int groupID, Integer learnerID) {
//			System.out.println("a learner of the group " + groupID + " is to be removed.");
//			System.out.println(this.history.get(groupID).toString());
//			System.out.println("learner ID to be removed: " + learnerID);
			this.history.get(groupID).remove(learnerID);
//			System.out.println(this.history.get(groupID).toString());
//			System.out.println("$$$$$$$$$");
		}
		
		public Collection<Concept> getConceptsFromGroup(int groupID){
			return this.get(groupID).values();
		}
		
		public ArrayList<Integer> getConceptKeySetFromGroup(int groupID){
			return this.get(groupID).keys();
		}
		
		public HashMap<Integer, Double> getConceptAgesFromGroup(int groupID){
			HashMap<Integer, Double> conceptAgesMapping = new HashMap<Integer, Double>();
			for (int key: this.get(groupID).keys()) {
				conceptAgesMapping.put(key, (double) this.get(groupID).getConcept(key).getCreatedOn());
			} 
//			System.out.println("Ages (createdOn) Mapping");
//			System.out.println(conceptAgesMapping.toString());
			return conceptAgesMapping;
		}
		
		public HashMap<Integer, Double> getConceptAgesFromGroupSubset(int groupID, ArrayList<Integer> subset){
			System.out.println("Subset of classfiers is: " + subset.toString());
			HashMap<Integer, Double> conceptAgesMapping = new HashMap<Integer, Double>();
			for (int key: this.get(groupID).keys()) {
				if (subset.contains(key)) {
					conceptAgesMapping.put(key, (double) this.get(groupID).getConcept(key).getCreatedOn());
				}
			} 
			System.out.println("Ages (createdOn) Mapping for a subset of classifiers");
			System.out.println(conceptAgesMapping.toString());
			return conceptAgesMapping;
		}
		
		public HashMap<Integer, Double> getConceptTimesUsedFromGroup(int groupID){
			HashMap<Integer, Double> conceptTimesUsedMapping = new HashMap<Integer, Double>();
			for (int key: this.get(groupID).keys()) {
				conceptTimesUsedMapping.put(key, (double) this.get(groupID).getConcept(key).getTimesUsed());
			} 
//			System.out.println("Times Reused Mapping");
//			System.out.println(conceptTimesUsedMapping.toString());
			return conceptTimesUsedMapping;
		}
		
		/** Get number of examples trained in each classifierr*/
		public HashMap<Integer, Double> getConceptExamplesFromGroup(int groupID){
			HashMap<Integer, Double> conceptTimesUsedMapping = new HashMap<Integer, Double>();
			for (int key: this.get(groupID).keys()) {
				conceptTimesUsedMapping.put(key, (double) this.get(groupID).getConcept(key).getNumberOfTrainingExamples());
			} 
//			System.out.println("Times Reused Mapping");
//			System.out.println(conceptTimesUsedMapping.toString());
			return conceptTimesUsedMapping;
		}
		
		
		
		
		public HashMap<Integer, Double> getConceptLastTimeUsedFromGroup(int groupID){
			HashMap<Integer, Double> conceptLastTimeUsed = new HashMap<Integer, Double>();
			for (int key: this.get(groupID).keys()) {
				conceptLastTimeUsed.put(key, (double) this.get(groupID).getConcept(key).instancesSeen);
			} 
//			System.out.println("Last Time Reused (instancesSeen) Mapping");
//			System.out.println(conceptLastTimeUsed.toString());
			return conceptLastTimeUsed;
		}
		
		public HashMap<Integer, Double> getConceptLastTimeUsedFromGroupSubset(int groupID, ArrayList<Integer> subset){
			System.out.println("Subset of classfiers is: " + subset.toString());
			HashMap<Integer, Double> conceptLastTimeUsed = new HashMap<Integer, Double>();
			for (int key: this.get(groupID).keys()) {
				if (subset.contains(key)) {
					conceptLastTimeUsed.put(key, (double) this.get(groupID).getConcept(key).instancesSeen);
				}
			} 
			System.out.println("Last Time Reused (instancesSeen) Mapping");
			System.out.println(conceptLastTimeUsed.toString());
			return conceptLastTimeUsed;
		}				
	}
	

	/** Window-related parameters for classifier internal comparisons during the warning window */
	protected class Window {
		int windowSize;
		int windowDefaultSize;
		int windowIncrements;
		int minWindowSize;
		int windowResizePolicy;
		double decisionThreshold;
		boolean rememberWindowSize;
		boolean backgroundDynamicWindowsFlag;

		public Window(int windowSize, int windowIncrements, int minWindowSize, double decisionThreshold,
				boolean rememberWindowSize, boolean backgroundDynamicWindowsFlag, int windowResizePolicy) {
			this.windowSize = windowSize;
			// the default size of a window could change overtime if there is window size inheritance enabled
			this.windowDefaultSize = windowSize;
			this.windowIncrements = windowIncrements;
			this.minWindowSize = minWindowSize;
			this.decisionThreshold = decisionThreshold;
			this.backgroundDynamicWindowsFlag = backgroundDynamicWindowsFlag;
			this.windowResizePolicy = windowResizePolicy;
			this.rememberWindowSize = rememberWindowSize;
		}

		public Window copy() {
			return new Window(this.windowSize, this.windowIncrements, this.minWindowSize, this.decisionThreshold,
					this.rememberWindowSize, this.backgroundDynamicWindowsFlag, this.windowResizePolicy);
		}

		public int getSize() {
			return this.windowSize;
		}

		public void setSize(int windowSize) {
			this.windowSize = windowSize;
		}

		public int getIncrements() {
			return this.windowIncrements;
		}

		public void setIncrements(int windowIncrements) {
			this.windowIncrements = windowIncrements;
		}

		public int getDefaultSize() {
			return this.windowDefaultSize;
		}

		public void setDefaultSize(int windowDefaultSize) {
			this.windowDefaultSize = windowDefaultSize;
		}

		public int getMinSize() {
			return this.minWindowSize;
		}

		public void setMinSize(int minWindowSize) {
			this.minWindowSize = minWindowSize;
		}

		public double getDecisionThreshold() {
			return this.decisionThreshold;
		}

		public void setDecisionThreshold(double decisionThreshold) {
			this.decisionThreshold = decisionThreshold;
		}

		public boolean getRememberSizeFlag() {
			return this.rememberWindowSize;
		}

		public void setRememberSizeFlag(boolean flag) {
			this.rememberWindowSize = flag;
		}

		public int getResizingPolicy() {
			return this.windowResizePolicy;
		}

		public void setResizingPolicy(int value) {
			this.windowResizePolicy = value;
		}

		public boolean getDynamicWindowInOldClassifiersFlag() {
			return this.backgroundDynamicWindowsFlag;
		}

		public void getDynamicWindowInOldClassifiersFlag(boolean flag) {
			this.backgroundDynamicWindowsFlag = flag;
		}

	}

	/** Object for events so the code is cleaner */
	public class Event {
		String instanceNumber;
		String event;
		String affectedPosition;
		String votingWeigth;
		String warningSetting;
		String driftSetting;
		String createdOn;
		String lastError;
		String numberOfClassifiers;
		String numberOfActiveWarnings;
		String classifiersOnWarning;
		String listOfApplicableConcepts;
		String recurringDriftToClassifierID;
		String driftToClassifierCreatedOn;
		String topologyExamples;
		String topologyPrototypes;
		String topologyError;

		// Constructor from array
		public Event(String[] eventDetails) {
			instanceNumber = eventDetails[0];
			event = eventDetails[1];
			affectedPosition = eventDetails[2];
			votingWeigth = eventDetails[3];
			warningSetting = eventDetails[4];
			driftSetting = eventDetails[5];
			createdOn = eventDetails[6];
			lastError = eventDetails[7];
			numberOfClassifiers = eventDetails[8];
			numberOfActiveWarnings = eventDetails[9];
			classifiersOnWarning = eventDetails[10];
			listOfApplicableConcepts = eventDetails[11];
			recurringDriftToClassifierID = eventDetails[12];
			driftToClassifierCreatedOn = eventDetails[13];
			topologyExamples = eventDetails[14];
			topologyPrototypes = eventDetails[15];
			topologyError = eventDetails[16];
		}

		// Getters and setters

		public String getInstanceNumber() {
			return instanceNumber;
		}

		public void setInstanceNumber(String instanceNumber) {
			this.instanceNumber = instanceNumber;
		}

		public String getEvent() {
			return event;
		}

		public void setEvent(String event) {
			this.event = event;
		}

		public String getAffectedPosition() {
			return affectedPosition;
		}

		public void setAffectedPosition(String affectedPosition) {
			this.affectedPosition = affectedPosition;
		}

		public String getVotingWeigth() {
			return votingWeigth;
		}

		public void setVotingWeigth(String votingWeigth) {
			this.votingWeigth = votingWeigth;
		}

		public String getWarningSetting() {
			return warningSetting;
		}

		public void setWarningSetting(String warningSetting) {
			this.warningSetting = warningSetting;
		}

		public String getDriftSetting() {
			return driftSetting;
		}

		public void setDriftSetting(String driftSetting) {
			this.driftSetting = driftSetting;
		}

		public String getCreatedOn() {
			return createdOn;
		}

		public void setCreatedOn(String createdOn) {
			this.createdOn = createdOn;
		}

		public String getLastError() {
			return lastError;
		}

		public void setLastError(String lastError) {
			this.lastError = lastError;
		}
		
		public String getTopologyError() {
			return this.topologyError;
		}

		public void setTopologyError(String topologyError) {
			this.topologyError = topologyError;
		}
		
		public String getTopologyExamples() {
			return this.topologyExamples;
		}

		public void setTopologyExamples(String topologyExamples) {
			this.topologyExamples = topologyExamples;
		}
		
		public String getTopologyPrototypes() {
			return this.topologyPrototypes;
		}

		public void setTopologyPrototypes(String topologyPrototypes) {
			this.topologyPrototypes = topologyPrototypes;
		}

		public String getNumberOfClassifiers() {
			return numberOfClassifiers;
		}

		public void setNumberOfClassifiers(String numberOfClassifiers) {
			this.numberOfClassifiers = numberOfClassifiers;
		}

		public String getNumberOfActiveWarnings() {
			return numberOfActiveWarnings;
		}

		public void setNumberOfActiveWarnings(String numberOfActiveWarnings) {
			this.numberOfActiveWarnings = numberOfActiveWarnings;
		}

		public String getClassifiersOnWarning() {
			return classifiersOnWarning;
		}

		public void setClassifiersOnWarning(String classifiersOnWarning) {
			this.classifiersOnWarning = classifiersOnWarning;
		}

		public String getListOfApplicableConcepts() {
			return listOfApplicableConcepts;
		}

		public void setListOfApplicableConcepts(String listOfApplicableConcepts) {
			this.listOfApplicableConcepts = listOfApplicableConcepts;
		}

		public String getRecurringDriftToClassifierID() {
			return recurringDriftToClassifierID;
		}

		public void setRecurringDriftToClassifierID(String recurringDriftToClassifierID) {
			this.recurringDriftToClassifierID = recurringDriftToClassifierID;
		}

		public String getDriftCreation() {
			return driftToClassifierCreatedOn;
		}

		public void setDriftToClassifierCreatedOn(String driftToClassifierCreatedOn) {
			this.driftToClassifierCreatedOn = driftToClassifierCreatedOn;
		}
		
		

	}
}

