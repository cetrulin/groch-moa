/*
 *    RecurringConceptsAdaptiveRandomForest.java
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

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.github.javacliparser.StringOption;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;

import moa.classifiers.trees.ARFHoeffdingTree;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.DynamicWindowClassificationPerformanceEvaluator;
import moa.evaluation.LearningPerformanceEvaluator;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import moa.classifiers.core.driftdetection.ChangeDetector;


/**
 * Recurring Concepts Adaptive Random Forest
 *
 * <p>Originally from Adaptive Random Forest (ARF). The 3 most important aspects of this 
 * ensemble classifier are: (1) inducing diversity through resampling;
 * (2) inducing diversity through randomly selecting subsets of features for 
 * node splits (See moa.classifiers.trees.ARFHoeffdingTree.java); (3) drift 
 * detectors per base tree, which cause selective resets in response to drifts. 
 * It also allows training background trees, which start training if a warning
 * is detected and replace the active tree if the warning escalates to a drift. </p>
 *
 * <p>See details in:<br> Heitor Murilo Gomes, Albert Bifet, Jesse Read, 
 * Jean Paul Barddal, Fabricio Enembreck, Bernhard Pfharinger, Geoff Holmes, 
 * Talel Abdessalem. Adaptive random forests for evolving data stream classification. 
 * In Machine Learning, DOI: 10.1007/s10994-017-5642-8, Springer, 2017.</p>
 *
 * <p>Parameters:</p> <ul>
 * <li>-l : Classiï¬�er to train. Must be set to ARFHoeffdingTree</li>
 * <li>-s : The number of trees in the ensemble</li>
 * <li>-o : How the number of features is interpreted (4 options): 
 * "Specified m (integer value)", "sqrt(M)+1", "M-(sqrt(M)+1)"</li>
 * <li>-m : Number of features allowed considered for each split. Negative 
 * values corresponds to M - m</li>
 * <li>-a : The lambda value for bagging (lambda=6 corresponds to levBag)</li>
 * <li>-j : Number of threads to be used for training</li>
 * <li>-x : Change detector for drifts and its parameters</li>
 * <li>-p : Change detector for warnings (start training bkg learner)</li>
 * <li>-w : Should use weighted voting?</li>
 * <li>-u : Should use drift detection? If disabled then bkg learner is also disabled</li>
 * <li>-q : Should use bkg learner? If disabled then reset tree immediately</li>
 * </ul>
 *
 * @author Andres Leon Suarez Cetrulo (suarezcetrulo at gmail dot com)
 * @version $Revision: 1 $
 */
public class RCARF extends AbstractClassifier implements MultiClassClassifier {

    @Override
    public String getPurposeString() {
        return "Recurring Concepts Adaptive Random Forest algorithm for evolving data streams from Suarez-Cetrulo et al.";
    }
    
    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.ARFHoeffdingTree -e 2000000 -g 50 -c 0.01");
    
    /*public ClassOption treeLearnerOption = new ClassOption("treeLearner", 'l',
            "Random Forest Tree.", ARFHoeffdingTree.class,
            "ARFHoeffdingTree -e 2000000 -g 50 -c 0.01");*/

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
        "The number of trees.", 10, 1, Integer.MAX_VALUE);
    
    public MultiChoiceOption mFeaturesModeOption = new MultiChoiceOption("mFeaturesMode", 'o', 
        "Defines how m, defined by mFeaturesPerTreeSize, is interpreted. M represents the total number of features. (Only for Adaptive Random Forest Hoeffding Tree)",
        new String[]{"Specified m (integer value)", "sqrt(M)+1", "M-(sqrt(M)+1)",
            "Percentage (M * (m / 100))"},
        new String[]{"SpecifiedM", "SqrtM1", "MSqrtM1", "Percentage"}, 1);
    
    public IntOption mFeaturesPerTreeSizeOption = new IntOption("mFeaturesPerTreeSize", 'm',
        "Number of features allowed considered for each split. Negative values corresponds to M - m. (Only for Adaptive Random Forest Hoeffding Tree)", 2, Integer.MIN_VALUE, Integer.MAX_VALUE);
    
    public FloatOption lambdaOption = new FloatOption("lambda", 'a',
        "The lambda parameter for bagging.", 6.0, 1.0, Float.MAX_VALUE);

    public IntOption numberOfJobsOption = new IntOption("numberOfJobs", 'j',
        "Total number of concurrent jobs used for processing (-1 = as much as possible, 0 = do not use multithreading)", 1, -1, Integer.MAX_VALUE);
    
    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
        "Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-5");

    public ClassOption warningDetectionMethodOption = new ClassOption("warningDetectionMethod", 'p',
        "Change detector for warnings (start training bkg learner)", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-4");
    
    public FlagOption disableWeightedVote = new FlagOption("disableWeightedVote", 'w', 
            "Should use weighted voting?");
    
    public FlagOption disableDriftDetectionOption = new FlagOption("disableDriftDetection", 'u',
        "Should use drift detection? If disabled then bkg learner is also disabled");

    public FlagOption disableBackgroundLearnerOption = new FlagOption("disableBackgroundLearner", 'q', 
        "Should use bkg learner? If disabled then reset tree immediately.");
    
	// ////////////////////////////////////////////////
	// ADDED IN RCARF by @suarezcetrulo
	// ////////////////////////////////////////////////
    public FlagOption disableRecurringDriftDetectionOption = new FlagOption("disableRecurringDriftDetection", 'r', 
            "Should save old learners to compare against in the future? If disabled then recurring concepts are not handled explicitely.");
    
    public FlagOption rememberConceptWindowOption = new FlagOption("rememberConceptWindow", 'i', 
            "Should remember last window size when retrieving a concept? If disabled then retrieved concepts will have a default window size.");
    
    public IntOption defaultWindowOption = new IntOption("defaultWindow", 'd', 
            "Number of rows by default in Dynamic Sliding Windows.", 10, 1, Integer.MAX_VALUE);
    
    public IntOption windowIncrementsOption = new IntOption("windowIncrements", 'c', 
            "Size of the increments or decrements in Dynamic Sliding Windows.", 1, 1, Integer.MAX_VALUE);
    
    public IntOption minWindowSizeOption = new IntOption("minWindowSize", 'z', 
            "Minimum window size in Dynamic Sliding Windows.", 5, 1, Integer.MAX_VALUE);
    
    public IntOption windowResizePolicyOption = new IntOption("windowResizePolicy",'y', 
    		"Policy to update the size of the window. Ordered by complexity, being 0 the simplest one and 3 the one with most complexity.", 0, 0, 2);
    
    public FloatOption thresholdOption = new FloatOption("thresholdOption", 't', 
            "Decision threshold for recurring concepts (-1 = threshold option disabled).", 0.65, -1, Float.MAX_VALUE);
    
    public FlagOption resizeAllWindowsOption = new FlagOption("resizeAllWindows", 'b', 
    		"Should the comparison windows for old learners be also dynamic?");
    		//+ "(0 = only the active model has a dynamic window, 1 = active and background models have dynamic windows, 2 = all models, "
    		//+ "including historic concepts). Window size changes in historic concepts during evaluation will only be saved "
    		//+ "if the historic model is selected as new active model and the threshold option is not disabled.", 1, 0, 2);
    
    public StringOption eventsLogFileOption = new StringOption("eventsLogFile",'e',"File path to export events as warnings and drifts", "./RCARF_events_log.txt");
    
    public FlagOption disableEventsLogFileOption = new FlagOption("disableEventsLogFile", 'g', 
            "Should export event logs to analyze them in the future? If disabled then events are not logged.");
   
    public IntOption logLevelOption = new IntOption("eventsLogFileLevel", 'h', 
            "0 only logs drifts; 1 logs drifts + warnings; 2 logs every data example", 1, 0, 2);

    public ClassOption evaluatorOption = new ClassOption("baseClassifierEvaluator", 'f',
            "Classification performance evaluation method in each base classifier for voting.",
            LearningPerformanceEvaluator.class,
            "BasicClassificationPerformanceEvaluator");
    
    public IntOption driftDecisionMechanismOption = new IntOption("driftDecisionMechanism", 'k', 
            "0 does not take into account the performance active base classifier explicitely, at the time of the drift; 1 takes into consideration active classifiers", 
            0, 0, 2);

	// ////////////////////////////////////////////////
	// ////////////////////////////////////////////////
    protected static final int FEATURES_M = 0;
    protected static final int FEATURES_SQRT = 1;
    protected static final int FEATURES_SQRT_INV = 2;
    protected static final int FEATURES_PERCENT = 3;
    
    protected static final int SINGLE_THREAD = 0;
	
    protected RCARFBaseLearner[] ensemble;
    protected long instancesSeen;
    protected int subspaceSize;
    protected BasicClassificationPerformanceEvaluator evaluator;

    private ExecutorService executor;
    
    PrintWriter eventsLogFile;
    
    @Override
    public void resetLearningImpl() {
        // Reset attributes
        this.ensemble = null;
        this.subspaceSize = 0;
        this.instancesSeen = 0;
        this.evaluator = new BasicClassificationPerformanceEvaluator();
        
        // Multi-threading
        int numberOfJobs;
        if(this.numberOfJobsOption.getValue() == -1) 
            numberOfJobs = Runtime.getRuntime().availableProcessors();
        else 
            numberOfJobs = this.numberOfJobsOption.getValue();
        // SINGLE_THREAD and requesting for only 1 thread are equivalent. 
        // this.executor will be null and not used...
        if(numberOfJobs != RCARF.SINGLE_THREAD && numberOfJobs != 1)
            this.executor = Executors.newFixedThreadPool(numberOfJobs);
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
        ++this.instancesSeen;
        if(this.ensemble == null) 
            initEnsemble(instance);
        
        // 1 If the concept history is ready and it contains old models, testing in each old model internal evaluator (to compare against bkg one)
        if (!disableRecurringDriftDetectionOption.isSet() && ConceptHistory.historyList != null && ConceptHistory.modelsOnWarning.containsValue(true) && ConceptHistory.historyList.size() > 0) {
	        	for (Concept oldModel : ConceptHistory.historyList.values()) { // TODO: test this
	            DoubleVector oldModelVote = new DoubleVector(oldModel.ConceptLearner.getVotesForInstance(instance)); // TODO. this
	            // // System.out.println("Im classifier number #"+oldModel.Concept.classifier.calcByteSize()+" created on: "+oldModel.Concept.createdOn+"  and last error was:  "+oldModel.Concept.lastError);
        			if (oldModel.ConceptLearner.internalWindowEvaluator != null && 
        					oldModel.ConceptLearner.internalWindowEvaluator.getAmountOfApplicableModels() > 0) { // When the concept is added the first time, it doesn't have applicable models. They are not inserted until the first warning. 
        				// So the Concept History only runs over warning windows
        				oldModel.ConceptLearner.internalWindowEvaluator.addResult(new InstanceExample(instance), oldModelVote.getArrayRef()); // TODO: test this
        			}
	        	}
        } 
        
        Collection<TrainingRunnable> trainers = new ArrayList<TrainingRunnable>();
        for (int i = 0 ; i < this.ensemble.length ; i++) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(instance));
            InstanceExample example = new InstanceExample(instance);
            
            // 2 Testing in active model
            this.ensemble[i].evaluator.addResult(example, vote.getArrayRef());

            if(!disableRecurringDriftDetectionOption.isSet()) {
	            // 3 If the warning window is open, testing in background model internal evaluator (for comparison purposes) 
	            if(this.ensemble[i].bkgLearner != null && this.ensemble[i].bkgLearner.internalWindowEvaluator!=null 
	            		&& this.ensemble[i].bkgLearner.internalWindowEvaluator.containsIndex(this.ensemble[i].bkgLearner.indexOriginal)) {
	                 DoubleVector bkgVote = new DoubleVector(this.ensemble[i].bkgLearner.getVotesForInstance(instance)); 
	            		this.ensemble[i].bkgLearner.internalWindowEvaluator.addResult(example, bkgVote.getArrayRef());
	            }
            }
                        
            int k = MiscUtils.poisson(this.lambdaOption.getValue(), this.classifierRandom);
            if (k > 0) { // asuarez: this condition makes some trees not to be trained with some instances, improving diversity in the ensemble.
                if(this.executor != null) {
                    TrainingRunnable trainer = new TrainingRunnable(this.ensemble[i], 
                        instance, k, this.instancesSeen);  // asuarez: k is only an instance weight that increases when the instance is missclassified. this is a bagging strategy by Oza 2005.
                    trainers.add(trainer);
                }
                else { // SINGLE_THREAD is in-place... 
                    this.ensemble[i].trainOnInstance(instance, k, this.instancesSeen);
                }
            }
        }
        
        if(this.executor != null) {
            try {
                this.executor.invokeAll(trainers);
            } catch (InterruptedException ex) {
                throw new RuntimeException("Could not call invokeAll() on training threads.");
            }
        }
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        Instance testInstance = instance.copy();
        if(this.ensemble == null) 
            initEnsemble(testInstance);
        DoubleVector combinedVote = new DoubleVector();

        for(int i = 0 ; i < this.ensemble.length ; ++i) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(testInstance));
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                double acc = this.ensemble[i].evaluator.getPerformanceMeasurements()[1].getValue();
                if(! this.disableWeightedVote.isSet() && acc > 0.0) {                        
                    for(int v = 0 ; v < vote.numValues() ; ++v) {
                        vote.setValue(v, vote.getValue(v) * acc);
                    }
                }
                combinedVote.addValues(vote);
            }
        }
        return combinedVote.getArrayRef();
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
    		//eventsLogFile.close();
        return null;
    }

    
    protected void initEnsemble(Instance instance) {
    	
        // Init the ensemble.
        int ensembleSize = this.ensembleSizeOption.getValue();
        this.ensemble = new RCARFBaseLearner[ensembleSize];
        
        BasicClassificationPerformanceEvaluator classificationEvaluator = (BasicClassificationPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
        // OLD TODO: this should be an option with default = BasicClassificationPerformanceEvaluator
        // BasicClassificationPerformanceEvaluator classificationEvaluator = new BasicClassificationPerformanceEvaluator();
        
        this.subspaceSize = this.mFeaturesPerTreeSizeOption.getValue();
  
        // Only initialize Concept History if explicit recurring concepts handling is enabled
        if(!this.disableRecurringDriftDetectionOption.isSet()) {
			ConceptHistory.lastID = 0;
			ConceptHistory.historyList = new ConcurrentHashMap<Integer,Concept> ();
			ConceptHistory.modelsOnWarning = new ConcurrentHashMap<Integer,Boolean> ();
        }
        
        try { // Start events logging and print headers
	    		if (disableEventsLogFileOption.isSet()) {
	    			eventsLogFile = null;
	    		} else {
	    			eventsLogFile = new PrintWriter(this.eventsLogFileOption.getValue());
	    			logEvent(getEventHeaders()); 
	    		}
        	} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
        
        // The size of m depends on:
        // 1) mFeaturesPerTreeSizeOption
        // 2) mFeaturesModeOption
        int n = instance.numAttributes()-1; // Ignore class label ( -1 )
        
        switch(this.mFeaturesModeOption.getChosenIndex()) {
            case RCARF.FEATURES_SQRT:
                this.subspaceSize = (int) Math.round(Math.sqrt(n)) + 1;
                break;
            case RCARF.FEATURES_SQRT_INV:
                this.subspaceSize = n - (int) Math.round(Math.sqrt(n) + 1);
                break;
            case RCARF.FEATURES_PERCENT:
                // If subspaceSize is negative, then first find out the actual percent, i.e., 100% - m.
                double percent = this.subspaceSize < 0 ? (100 + this.subspaceSize)/100.0 : this.subspaceSize / 100.0;
                this.subspaceSize = (int) Math.round(n * percent);
                break;
        }
        // Notice that if the selected mFeaturesModeOption was 
        //  RecurringConceptsAdaptiveRandomForest.FEATURES_M then nothing is performed in the
        //  previous switch-case, still it is necessary to check (and adjusted) 
        //  for when a negative value was used. 
        
        // m is negative, use size(features) + -m
        if(this.subspaceSize < 0)
            this.subspaceSize = n + this.subspaceSize;
        // Other sanity checks to avoid runtime errors. 
        //  m <= 0 (m can be negative if this.subspace was negative and 
        //  abs(m) > n), then use m = 1
        if(this.subspaceSize <= 0)
            this.subspaceSize = 1;
        // m > n, then it should use n
        if(this.subspaceSize > n)
            this.subspaceSize = n;
        
        //ARFHoeffdingTree treeLearner = (ARFHoeffdingTree) getPreparedClassOption(this.treeLearnerOption);
        //treeLearner.resetLearning();        
        Classifier learner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        learner.resetLearning();
        
        for(int i = 0 ; i < ensembleSize ; ++i) {	
        	    // asuarez TO-DO: bagging should be in this code and not in the code of the trees for ARF. this is only a provisional fix.
        		if(learner.getPurposeString().contains("Adaptive Random Forest Hoeffding Tree for data streams.")) {
        			//System.out.println("The current base learner supports feature subspace. Applying it to classifier: #"+(i+1));
        			((ARFHoeffdingTree) learner).subspaceSizeOption.setValue(this.subspaceSize);
        		}
            this.ensemble[i] = new RCARFBaseLearner(
                i, 
                (Classifier) learner.copy(), 
                (BasicClassificationPerformanceEvaluator) classificationEvaluator.copy(), 
                this.instancesSeen, 
                ! this.disableBackgroundLearnerOption.isSet(),
                ! this.disableDriftDetectionOption.isSet(), 
                driftDecisionMechanismOption.getValue(),
                driftDetectionMethodOption,
                warningDetectionMethodOption,
                false,
                ! this.disableRecurringDriftDetectionOption.isSet(),
                false, // @suarezcetrulo : first model is not old. An old model (retrieved from the concept history).
                new Window(this.defaultWindowOption.getValue(), this.windowIncrementsOption.getValue(), this.minWindowSizeOption.getValue(), this.thresholdOption.getValue(), 
        				this.rememberConceptWindowOption.isSet()? true: false, this.resizeAllWindowsOption.isSet()? true: false, windowResizePolicyOption.getValue()),
                null, // @suarezcetrulo : Windows start at NULL
                eventsLogFile,
                logLevelOption.getValue()
            		);
        }
    }
    
    /**
     * Inner class that represents a single tree member of the forest. 
     * It contains some analysis information, such as the numberOfDriftsDetected, 
     */
    protected final class RCARFBaseLearner {
        public int indexOriginal;
        public long createdOn;
        public long lastDriftOn;
        public long lastWarningOn;
        public Classifier classifier;
        public boolean isBackgroundLearner;
        public boolean isOldLearner; // only for reference
        
        // The drift and warning object parameters. 
        protected ClassOption driftOption;
        protected ClassOption warningOption;
        
        // Drift and warning detection
        protected ChangeDetector driftDetectionMethod;
        protected ChangeDetector warningDetectionMethod;
        
        public boolean useBkgLearner;
        public boolean useDriftDetector;
        public boolean useRecurringLearner; // @suarezcetrulo
        public int driftDecisionMechanism; // @suarezcetrulo
        
        // Bkg learner
        protected RCARFBaseLearner bkgLearner;
        // Copy of main model at the beginning of the warning window for its copy in the Concept History
        protected Concept tmpCopyOfModel;  
        // Statistics
        public BasicClassificationPerformanceEvaluator evaluator;
        protected int numberOfDriftsDetected;
        protected int numberOfWarningsDetected;

        // Internal statistics
        public DynamicWindowClassificationPerformanceEvaluator internalWindowEvaluator; // only used in background and old classifiers
        protected double lastError;
        protected Window windowProperties;
        
        public PrintWriter eventsLogFile;
        public int logLevel;

        private void init(int indexOriginal, Classifier classifier, BasicClassificationPerformanceEvaluator evaluatorInstantiated, 
            long instancesSeen, boolean useBkgLearner, boolean useDriftDetector, int driftDecisionMechanism, ClassOption driftOption, ClassOption warningOption, boolean isBackgroundLearner, 
            boolean useRecurringLearner, boolean isOldLearner, Window windowProperties, DynamicWindowClassificationPerformanceEvaluator internalEvaluator,
            PrintWriter eventsLogFile, int logLevel) { // last parameters added by @suarezcetrulo
            this.indexOriginal = indexOriginal;
            this.createdOn = instancesSeen;
            this.lastDriftOn = 0;
            this.lastWarningOn = 0;
            this.eventsLogFile = eventsLogFile;
            this.logLevel = logLevel;
            
            this.classifier = classifier;
            this.evaluator = evaluatorInstantiated;
            this.useBkgLearner = useBkgLearner;
            this.useRecurringLearner = useRecurringLearner;
            this.useDriftDetector = useDriftDetector;
            this.driftDecisionMechanism = driftDecisionMechanism;
            
            this.numberOfDriftsDetected = 0;
            this.numberOfWarningsDetected = 0; // TODO. asuarez. it would be handy to track this parameter and the one above.
            this.isBackgroundLearner = isBackgroundLearner;
                        
            if(this.useDriftDetector) {
                this.driftOption = driftOption;
                this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftOption)).copy();
            }

            // Init Drift Detector for Warning detection. 
            if(this.useBkgLearner) {
                this.warningOption = warningOption;
                this.warningDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.warningOption)).copy();
            }       
            
            if (useRecurringLearner) {
                // Window params
                this.windowProperties = windowProperties;
                // Recurring drifts
                this.isOldLearner = isOldLearner;
                // only used in bkg and retrieved old models
                this.internalWindowEvaluator = internalEvaluator;
            }
        }

        // last inputs parameters added by @suarezcetrulo
        public RCARFBaseLearner(int indexOriginal, Classifier classifier, BasicClassificationPerformanceEvaluator evaluatorInstantiated, 
                    long instancesSeen, boolean useBkgLearner, boolean useDriftDetector, int driftDecisionMechanism, ClassOption driftOption, ClassOption warningOption, 
                    boolean isBackgroundLearner, boolean useRecurringLearner, boolean isOldLearner, 
                    Window windowProperties, DynamicWindowClassificationPerformanceEvaluator bkgInternalEvaluator, PrintWriter eventsLogFile, int logLevel) {
            init(indexOriginal, classifier, evaluatorInstantiated, instancesSeen, useBkgLearner, 
            		 useDriftDetector, driftDecisionMechanism, driftOption, warningOption, isBackgroundLearner, useRecurringLearner,  isOldLearner, 
            		 windowProperties,bkgInternalEvaluator, eventsLogFile, logLevel);
        }

        public void reset() {

            // System.out.println();
            // System.out.println("-------------------------------------------------");
            // System.out.println("RESET (WARNING OFF) IN MODEL #"+this.indexOriginal+". Warning flag status (activeModelPos, Flag): "+ConceptHistory.modelsOnWarning);
            // System.out.println("-------------------------------------------------");
            // System.out.println();
		    // Transition to the best bkg or retrieved old learner
        		if (this.useBkgLearner && this.bkgLearner != null) {
        		   if(this.useRecurringLearner) { // && ConceptHistory.historyList != null && ConceptHistory.historyList.size() > 0) {
           			// 1 Decrease amount of warnings in concept history and from evaluators
        				ConceptHistory.modelsOnWarning.put(this.indexOriginal, false);
        	            if(ConceptHistory.historyList != null && ConceptHistory.historyList.size() > 0) {
        			        	for (Concept oldModel : ConceptHistory.historyList.values()) {
        			        		if (oldModel.ConceptLearner.internalWindowEvaluator != null && 
        			        				oldModel.ConceptLearner.internalWindowEvaluator.containsIndex(this.indexOriginal) )
        				        		((DynamicWindowClassificationPerformanceEvaluator) 
        				        			oldModel.ConceptLearner.internalWindowEvaluator).deleteModel(this.indexOriginal);
        			        	}
        	            }
        			   // 2.1 Move copy of active model made before warning to Concept History. Its history ID will be the last one in the history (= size)
        			   // Clean the copy afterwards.
        			   this.tmpCopyOfModel.addHistoryID(ConceptHistory.nextID());
        			   ConceptHistory.historyList.put(this.tmpCopyOfModel.historyIndex, this.tmpCopyOfModel);
        			   this.tmpCopyOfModel = null;
        			   //// System.out.println("MODEL ADDED TO CONCEPT HISTORY!");
        			   // Consideration *: This classifier is added to the concept history, but it wont be considered by other classifiers on warning until their next warning.
        			   // If it becomes necessary in terms of implementation for this concept, to be considered immediately by the other examples in warning,
        			   // we could have a HashMap in ConceptHistory with a flag saying if a given ensembleIndexPos needs to check the ConceptHistory again and add window sizes and priorError.
                   
        			   // 2.2 Update window size in window properties depending on window size inheritance flag (entry parameter/Option)
       	           this.bkgLearner.windowProperties.setSize(((this.bkgLearner.windowProperties.rememberWindowSize) ? 
                   		this.bkgLearner.internalWindowEvaluator.getWindowSize(this.bkgLearner.indexOriginal) : this.bkgLearner.windowProperties.windowDefaultSize)); 
        		   
       	           // 2.3 Inherit window properties / clear internal evaluator
                   this.windowProperties = this.bkgLearner.windowProperties; // internalEvaluator shouldnt be inherited
                   this.internalWindowEvaluator = null; // only a double check, as it should be always null (only used in background + old concept Learners)

        		   }
	            // 2.3 New active model is the best retrieved old model / clear background learner
                this.classifier = this.bkgLearner.classifier;
                this.driftDetectionMethod = this.bkgLearner.driftDetectionMethod;
                this.warningDetectionMethod = this.bkgLearner.warningDetectionMethod;                
                this.evaluator = this.bkgLearner.evaluator;
                this.createdOn = this.bkgLearner.createdOn;
                this.bkgLearner = null; 
        		} 
            else { 
                this.classifier.resetLearning();
                this.createdOn = instancesSeen;
                this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftOption)).copy();
            }
            this.evaluator.reset();
        }

        public void trainOnInstance(Instance instance, double weight, long instancesSeen) {
            Instance weightedInstance = (Instance) instance.copy();
            weightedInstance.setWeight(instance.weight() * weight);
            // Training active models and background models (if they exist). Retrieved old models are not trained.
            this.classifier.trainOnInstance(weightedInstance);            
            if(this.bkgLearner != null) this.bkgLearner.classifier.trainOnInstance(instance);
            
            // Set false alarms in case of drift at false as default
            boolean falseAlarm = false; // Included for cases where driftDecisionMechanism > 0 and recurring drifts are enabled.
            
            // Should it use a drift detector? Also, is it a backgroundLearner? If so, then do not "incept" another one. 
            if(this.useDriftDetector && !this.isBackgroundLearner) {
                boolean correctlyClassifies = this.classifier.correctlyClassifies(instance);
                // Check for warning only if useBkgLearner is active
                if(this.useBkgLearner) { 
                    /*********** warning detection ***********/
                    // Update the WARNING detection method
                    this.warningDetectionMethod.input(correctlyClassifies ? 0 : 1);
                    // Check if there was a change – in case of false alarm this triggers warning again and the bkglearner gets replaced
                    if(this.warningDetectionMethod.getChange()) {
                        this.lastWarningOn = instancesSeen;
                        this.numberOfWarningsDetected++;

	    				   // 1 Update last error and make a backup of the current classifier in a concept object (the active one will be in use until the Drift is confirmed). 
                        // As there is no false alarms explicit mechanism (bkgLeaners keep running till replaced), this has been moved here.
                        if(this.useRecurringLearner) saveCurrentConcept();
	                	   
	                	   // 2 Start warning window to create bkg learner and retrieve old models (if option enabled)
                        startWarningWindow();
                    } 

                } /*********** drift detection ***********/
                // Update the DRIFT detection method
                this.driftDetectionMethod.input(correctlyClassifies ? 0 : 1);
                // Check if there was a change
                if(this.driftDetectionMethod.getChange()) {
                    this.lastDriftOn = instancesSeen;
                    this.numberOfDriftsDetected++;
                    //toLog = false;
                    
          		   // 1 Compare DT results using Window method and pick the best one between concept history and bkg model.
          		   // It returns the best model in the object of the bkgLearner
                    if (this.useRecurringLearner) falseAlarm = selectNextActiveModel(); // if there is not another base classifier with lower error than active model (and driftDecisionMechanism > 0), then a false alarm is raised
                    else if (eventsLogFile != null && logLevel >= 1 ) logEvent(getBkgDriftEvent()); // Print bkg drifts in log also for ARF

                    // 2 Transition to new model (only if there is no false alarms)
                    if (!falseAlarm) this.reset();
                    // asuarez. 25 sept 2018
                    // IMPORTANT. False alarms avoid drifts, but they do not disable a certain warning on a base classifier.
                    // In cases of a false alarm, the active classifiers will still have an active warning.
                } 
            } if (eventsLogFile != null && logLevel >= 2) logEvent(getTrainExampleEvent()); // Register training example in log
        } 
        
        // Saves a backup of the active model that raised a warning to be stored in the concept history in case of drift.
        public void saveCurrentConcept() {  
        		// if(ConceptHistory.historyList != null) // System.out.println("CONCEPT HISTORY SIZE IS: "+ConceptHistory.historyList.size());
        	
        		// 1 Update last error before warning of the active classifier
        		// This error is the total fraction of examples incorrectly classified since this model was active until now.
    			this.lastError = this.evaluator.getFractionIncorrectlyClassified();  
       			    			
    			// 2 Copy Base learner for Concept History in case of Drift and store it on temporal object.
    			// First, the internal evaluator will be null. 
    			// It doesn't get initialized till once in the Concept History and the first warning arises. See it in startWarningWindow
    			RCARFBaseLearner tmpConcept = new RCARFBaseLearner(this.indexOriginal, 
    					this.classifier.copy(), (BasicClassificationPerformanceEvaluator) this.evaluator.copy(), 
    					this.createdOn, this.useBkgLearner, this.useDriftDetector, this.driftDecisionMechanism, this.driftOption, this.warningOption, 
    					true, this.useRecurringLearner, true, this.windowProperties.copy(), null, eventsLogFile, logLevel);

    			this.tmpCopyOfModel = new Concept(tmpConcept, 
	    				this.createdOn, this.evaluator.getPerformanceMeasurements()[0].getValue(), this.lastWarningOn);
	    		
	    		// 3 Add the model accumulated error (from the start of the model) from the iteration before the warning
	    		this.tmpCopyOfModel.setErrorBeforeWarning(this.lastError);
	    		// A simple concept to be stored in the concept history that doesn't have a running learner.
	    		// This doesn't train. It keeps the model as it was at the beginning of the training window to be stored in case of drift.
        }
                
        // Starts Warning window
        public void startWarningWindow() {
        		// 0 Reset warning window
            	this.bkgLearner = null; 
    	        if(useRecurringLearner) {    	        	
	    	        	this.internalWindowEvaluator = null;
	    	        
	    	        // 1 Updating objects with warning. Turns on windows flag in Concept History.
	            // Also, if the concept history is ready and it contains old models, add prior estimation and window size to each concepts history learner
	        		ConceptHistory.modelsOnWarning.put(this.indexOriginal, true);
	            if(ConceptHistory.historyList != null && ConceptHistory.historyList.size() > 0) {
			        	for (Concept oldModel : ConceptHistory.historyList.values()) {
			        		// If the concept internal evaluator has been initialized for any other model on warning, add window size and last error of current model on warning 
			        		if (oldModel.ConceptLearner.internalWindowEvaluator != null) {
			        			//// System.out.println("ADDING VALUES TO INTERNAL EVALUATOR OF CONCEPT "+oldModel.historyIndex+" IN POS "+this.indexOriginal);
				        		((DynamicWindowClassificationPerformanceEvaluator) 
				        			oldModel.ConceptLearner.internalWindowEvaluator).addModel(this.indexOriginal,this.lastError,this.windowProperties.windowSize);
			        		} 
			        		// Otherwise, initialize a new internal evaluator for the concept
			        		else {
			        			//// System.out.println("INSTANCIATING FOR THE FIRST TIME INTERNAL EVALUATOR FOR CONCEPT "+oldModel.historyIndex+" IN POS "+this.indexOriginal);
		           			DynamicWindowClassificationPerformanceEvaluator tmpInternalWindow = new DynamicWindowClassificationPerformanceEvaluator(
		           				this.windowProperties.getSize(), this.windowProperties.getIncrements(), this.windowProperties.getMinSize(),
		                			this.lastError, this.windowProperties.getDecisionThreshold(),
		                			this.windowProperties.getDynamicWindowInOldModelsFlag(), this.windowProperties.getResizingPolicy(),
		                			this.indexOriginal, "created for old-retrieved classifier in ensembleIndex #"+this.indexOriginal);  	
		           			tmpInternalWindow.reset(); 
		           			
		           			oldModel.ConceptLearner.internalWindowEvaluator = tmpInternalWindow;
			        		}
			        	}
	            } 
    	        } 
	        	// Log warning     
    	        if (eventsLogFile != null && logLevel >= 1 ) logEvent(getWarningEvent());

            // 2 Create background Model
            createBkgModel();

            // Update the warning detection object for the current object 
            // (this effectively resets changes made to the object while it was still a bkg learner). 
            this.warningDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.warningOption)).copy();
        }
        
        // Creates BKG Model in warning window
        public void createBkgModel() {
        		// Empty explicitely the BKG internal evaluator if enabled
        		//if (this.bkgLearner != null) this.bkgLearner.internalWindowEvaluator.clear(); // I don't see any improvement
        	
            // 1 Create a new bkgTree classifier
            Classifier bkgClassifier = this.classifier.copy();
            bkgClassifier.resetLearning();
                                    
            // 2 Resets the evaluator
            BasicClassificationPerformanceEvaluator bkgEvaluator = (BasicClassificationPerformanceEvaluator) this.evaluator.copy();
            bkgEvaluator.reset();
            
            // // System.out.println("------------------------------");
            // // System.out.println("Create estimator for BKG model in position: "+this.indexOriginal);
            // 3 Adding also internal evaluator (window) in bkgEvaluator (by @suarezcetrulo)
            DynamicWindowClassificationPerformanceEvaluator bkgInternalWindowEvaluator = null;
            if(this.useRecurringLearner) {
                bkgInternalWindowEvaluator = new DynamicWindowClassificationPerformanceEvaluator (
                		this.windowProperties.getSize(),this.windowProperties.getIncrements(),this.windowProperties.getMinSize(),
                		this.lastError,this.windowProperties.getDecisionThreshold(),true,this.windowProperties.getResizingPolicy(), 
                		this.indexOriginal, "created for BKG classifier in ensembleIndex #"+this.indexOriginal);  
                bkgInternalWindowEvaluator.reset();
            }
            // // System.out.println("------------------------------");
            
            // 4 Create a new bkgLearner object
            this.bkgLearner = new RCARFBaseLearner(indexOriginal, bkgClassifier, bkgEvaluator, this.lastWarningOn, 
            		this.useBkgLearner, this.useDriftDetector, this.driftDecisionMechanism, this.driftOption, this.warningOption, true, this.useRecurringLearner, false, 
            									   this.windowProperties, bkgInternalWindowEvaluator, eventsLogFile, logLevel); // added last inputs parameter by @suarezcetrulo        	
        }
        
        /** 
         *  This method ranks all applicable base classifiers in the Concept History (CH)
         *  It also selects the next model to be active, or it raises a false alarm if the drift should be reconsidered.
         * 
		 *  -----------------------------------------------------------------------------------------------------------------
         *  False alarms depend on the drift decision mechanism
		 *  -----------------------------------------------------------------------------------------------------------------
		 *  
		 * 	When driftDecisionMechanism == 0, if bkgLearner == null, false alarms cannot be raised. A comparison against CH is not possible as there is no bkg learner trained.
	     *	 In this case, a drift signal has been raised and it cannot be stopped without false alarms. A bkg drift applies as only option available.
		 *  
         * 	When drift decision mechanism == 1 or 2, then false alarms are taken into consideration for drifts (the warning will be still active even if a false alarm is raised for a drift in the same active model).
	    	 *	If the background learner is NULL, we consider that the drift signal may have been caused by a too sensitive drift detection parameterization.
	    	 *	In this case, it's clearly too soon to change the active model. Therefore we raise a drift signal. 
	     *		
	     *	When drift decision mechanism == 2, we also raise a false alarm when the active model obtains less error than the bkg classifier and all of the classifiers from the CH.
	     *		
	     *  -----------------------------------------------------------------------------------------------------------------
		 *	If the active classifier is not the best available choice / false alarm is raised, the following logic applies:
		 *  -----------------------------------------------------------------------------------------------------------------
		 *  If bkgBetterThanCHbaseClassifier == False, the minimum error of the base classifiers in the CH is not lower than the error of the bkg classifier. 
		 *   Then, register background drift.
		 *   
		 *	If CHranking.size() == 0, no applicable concepts for the active model in the concept history. Then, we register background drift.
	    	 *	Otherwise, a recurring drift is the best option. 
	    	 *			    		 
         * */
        public boolean selectNextActiveModel() {
        		// traces for testing asuarez 28/10/2018
        		/*System.out.println("---------------------------------------------");
        		System.out.println(instancesSeen + " - Drift signal at tree #"+this.indexOriginal);*/
	    		// 1 Raise a false alarm for the drift if the background learner is not ready
	    		if (this.driftDecisionMechanism > 0 && this.bkgLearner == null) {
	    			// traces for testing asuarez 28/10/2018
	    			/*System.out.println("RESULT: false alarm - background learner not ready");*/
	    			return registerDriftFalseAlarm();}
	        	
	    		// 2 Retrieve best applicable model from Concept History
        		int indexOfBestRanked = -1;
        		double errorOfBestRanked = -1.0;
        		HashMap<Integer, Double> ranking = rankConceptHistoryClassifiers();
	    		if (ranking.size() > 0) {
	    			indexOfBestRanked = getMinKey(ranking); // find index of concept with lowest value (error)
	    			errorOfBestRanked = Collections.min(ranking.values());
	    		}
        		
	    		// 3 Compare this against the background model and make the decision.
	    		if (this.driftDecisionMechanism == 2) {
    				if (activeBetterThanBKGbaseClassifier()) { 
    					if (ranking.size() >0 && !activeBetterThanCHbaseClassifier(errorOfBestRanked))
		    	    				 registerRecurringDrift(indexOfBestRanked); 
    						// false alarm if active model is still the best one and when there are no applicable concepts.
		    				else {
		    					// traces for testing asuarez 28/10/2018
		    					/*System.out.println("RESULT: false alarm - active model is best");*/
		    						return registerDriftFalseAlarm(); }
    				} else { 
    			    		if(ranking.size() > 0 && bkgBetterThanCHbaseClassifier(errorOfBestRanked)) 
    			    			registerRecurringDrift(indexOfBestRanked);
    			    			
    			    		else registerBkgDrift (); 					
    				}
    				
    			// Drift decision mechanism == 0 or 1 (in an edge case where the bkgmodel is still NULL, we ignore the comparisons)
	    		} else {
	    			if(ranking.size() > 0 && this.bkgLearner != null && bkgBetterThanCHbaseClassifier(errorOfBestRanked)) 
	    				registerRecurringDrift (indexOfBestRanked);
			    	else registerBkgDrift ();
	    		} 
	    		// traces for testing asuarez 28/10/2018
			/*
	    		System.out.println("RESULT: Drift");
        		System.out.println("---------------------------------------------");*/

	    		return false; // No false alarms raised at this point
        }
        
        /*private HashMap<Integer, Double> updateWithExistingConcepts(HashMap<Integer, Double> ranking) {
	    		if(ranking.size()>0) {
	    			int min = getMinKey(ranking);
	    			if(!ConceptHistory.historyList.containsKey(min)) {
	    				ranking.remove(min);
	    				if(ranking.size()>0) ranking = updateWithExistingConcepts(ranking);
		    		}
	    		} return ranking;
        }*/

        // this.indexOriginal - pos of this model with active warning in ensemble
        public HashMap<Integer, Double> rankConceptHistoryClassifiers () {
	    		HashMap<Integer, Double> CHranking = new HashMap<Integer, Double>();
	    		// Concept History owns only one learner per historic concept. But each learner saves all model's independent window size and priorEstimation in a HashMap.
	    		for (Concept auxConcept : ConceptHistory.historyList.values()) 
	    			// Only take into consideration Concepts sent to the Concept History after the current model raised a warning (see this consideration in reset*) 
	    			if (auxConcept.ConceptLearner.internalWindowEvaluator != null && 
	    					auxConcept.ConceptLearner.internalWindowEvaluator.containsIndex(this.indexOriginal)) { // checking indexOriginal to verify that it's an applicable concept
		    			CHranking.put(auxConcept.getHistoryIndex(), ((DynamicWindowClassificationPerformanceEvaluator) 
		    					auxConcept.ConceptLearner.internalWindowEvaluator).getFractionIncorrectlyClassified(this.indexOriginal)); // indexOriginal refers to the model we compare against. it should be an applicable concept.
	    			}
	    		
	    		// Double check just in case the best concept is no longer in the concept history, or some concepts where created after the current model raised warning.
	    		// CHranking = updateWithExistingConcepts(ranking);
	    		return CHranking;
        }
        
        // Aux method for getting the best classifier in a hashMap of (int modelIndex, double averageErrorInWindow) 
        private Integer getMinKey(Map<Integer, Double> map) {
        	    Integer minKey = null;
        	    // System.out.println("map is: "+map+" number of keys is: "+map.keySet().size()); // TODO. debugging
            double minValue = Double.MAX_VALUE;
            for(Integer key : map.keySet()) {
            		// System.out.println("Key is:"+key+" with value: "+map.get(key));
                 double value = map.get(key);
                 if(value < minValue) {
                	   // System.out.println("Min error is: "+ value+" with key: "+key);
                    minValue = value;
                    minKey = key;
                }
            } return minKey;
        }
        
        // this.indexOriginal - pos of this model with active warning in ensemble
        public boolean activeBetterThanBKGbaseClassifier() { 
	        	// TODO? We may need to do use an internal evaluator for the active learner when driftDecisionMechanism==2. 
	        	//		 But the resizing mechanism may need to be different, or compare to the bkg learner.
	        	/* return (((DynamicWindowClassificationPerformanceEvaluator) this.internalWindowEvaluator)
	        				.getFractionIncorrectlyClassified(this.indexOriginal) <= 
	        			((DynamicWindowClassificationPerformanceEvaluator) this.bkgLearner.internalWindowEvaluator)
	        				.getFractionIncorrectlyClassified(this.bkgLearner.indexOriginal));*/
        		
        		// traces for testing asuarez 28/10/2018
			/*System.out.println("-");
			System.out.println("Active has an error of: "+this.evaluator.getFractionIncorrectlyClassified()+" while BKG has an error of: "+this.evaluator.getFractionIncorrectlyClassified());
			System.out.println("So it returns: "+(this.evaluator.getFractionIncorrectlyClassified() <= 
        			((DynamicWindowClassificationPerformanceEvaluator) this.bkgLearner.internalWindowEvaluator)
        			.getFractionIncorrectlyClassified(this.bkgLearner.indexOriginal))+" TRUE means that active tree is better than BKG tree");
    			System.out.println("-");*/

	        return (( this.evaluator.getFractionIncorrectlyClassified() <= 
	        			((DynamicWindowClassificationPerformanceEvaluator) this.bkgLearner.internalWindowEvaluator)
	        			.getFractionIncorrectlyClassified(this.bkgLearner.indexOriginal))); 
	        // this.bkgLearner.indexOriginal == this.indexOriginal, as it is created for that same ensemble pos.  
        }
        
        // this.indexOriginal - pos of this model with active warning in ensemble
		public boolean activeBetterThanCHbaseClassifier(double bestFromCH) {			
	        	// TODO? We may need to do use an internal evaluator for the active learner when driftDecisionMechanism==2. 
	        	//		 But the resizing mechanism may need to be different, or compare to the bkg learner.
	        	/* return (((DynamicWindowClassificationPerformanceEvaluator) this.internalWindowEvaluator)
						.getFractionIncorrectlyClassified(this.indexOriginal) <= bestFromCH);*/
			
			// traces for testing asuarez 28/10/2018
			/*
			System.out.println("Active has an error of: "+this.evaluator.getFractionIncorrectlyClassified()+" while the best from CH has an error of: "+bestFromCH);
			System.out.println("So it returns: "+(this.evaluator.getFractionIncorrectlyClassified() <= bestFromCH)+" TRUE means that active tree is better than CH tree");
			*/
			return (this.evaluator.getFractionIncorrectlyClassified() <= bestFromCH);
		}

		// this.bkgLearner.indexOriginal - pos of bkg model if it becomes active in the ensemble
		public boolean bkgBetterThanCHbaseClassifier(double bestFromCH) {
			return (bestFromCH <= ((DynamicWindowClassificationPerformanceEvaluator) this.bkgLearner.internalWindowEvaluator)
					.getFractionIncorrectlyClassified(this.bkgLearner.indexOriginal));
	        // this.bkgLearner.indexOriginal == this.indexOriginal, as it is created for that same ensemble pos.  
		}

        public boolean registerDriftFalseAlarm () {
			// Register false alarm.
			if (this.eventsLogFile != null && this.logLevel >= 0 ) logEvent(getFalseAlarmEvent());	
			
			// Update false alarm (the active model will remain being the same)
			return true;
        }
        
        public void registerRecurringDrift (Integer indexOfBestRanked) {
			// Register recurring drift
			if (this.eventsLogFile != null && this.logLevel >= 0 ) logEvent(getRecurringDriftEvent(indexOfBestRanked));	
			
			// Extracts best recurring learner from concept history. It no longer exists in the concept history
            this.bkgLearner = ConceptHistory.extractConcept(indexOfBestRanked);       	
        }
        
        public void registerBkgDrift () {
			// Register background drift
			if (this.eventsLogFile != null && this.logLevel >= 0 ) logEvent(getBkgDriftEvent());	  
        }
        // ////////////////
        
        
        
        public double[] getVotesForInstance(Instance instance) {
            DoubleVector vote = new DoubleVector(this.classifier.getVotesForInstance(instance));
            return vote.getArrayRef();
        }
        
        // Auxiliar methods for logging events
           
        public Event getTrainExampleEvent() {
        		String [] eventLog = {
		        String.valueOf(instancesSeen), "Train example", String.valueOf(this.indexOriginal), 
				String.valueOf(this.evaluator.getPerformanceMeasurements()[1].getValue()), 
				this.warningOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), 
				this.driftOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), 
				String.valueOf(this.createdOn), String.valueOf(this.evaluator.getFractionIncorrectlyClassified()), 
				String.valueOf(this.useRecurringLearner ? ConceptHistory.modelsOnWarning.size() : "N/A"),
				String.valueOf(this.useRecurringLearner ? ConceptHistory.getNumberOfActiveWarnings() : "N/A"),
				String.valueOf(this.useRecurringLearner ? ConceptHistory.modelsOnWarning : "N/A"), "N/A", "N/A", "N/A"};
        		
        		return (new Event(eventLog));
        }
           
        public Event getWarningEvent() {
        	
            // System.out.println();
            // System.out.println("-------------------------------------------------");
            // System.out.println("WARNING ON IN MODEL #"+this.indexOriginal+". Warning flag status (activeModelPos, Flag): "+ConceptHistory.modelsOnWarning);
            // System.out.println("CONCEPT HISTORY STATE AND APPLICABLE FROM THIS WARNING IS: "+ConceptHistory.historyList.keySet().toString());
            // System.out.println("-------------------------------------------------");
            // System.out.println();
        	
        		String [] warningLog = {
    				String.valueOf(this.lastWarningOn), "WARNING-START", // event
    				String.valueOf(this.indexOriginal), String.valueOf(this.evaluator.getPerformanceMeasurements()[1].getValue()), 
    				this.warningOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
    				this.driftOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), 
    				String.valueOf(this.createdOn), String.valueOf(this.evaluator.getFractionIncorrectlyClassified()), 
    				String.valueOf(this.useRecurringLearner ? ConceptHistory.modelsOnWarning.size() : "N/A"),
    				String.valueOf(this.useRecurringLearner ? ConceptHistory.getNumberOfActiveWarnings() : "N/A"),
    				String.valueOf(this.useRecurringLearner ? ConceptHistory.modelsOnWarning : "N/A"), 
    				this.useRecurringLearner ? ConceptHistory.historyList.keySet().toString() : "N/A", "N/A", "N/A"};
        		//1279,1,WARNING-START,0.74,{F,T,F;F;F;F},...

        		return (new Event(warningLog));
        }
        
        public Event getBkgDriftEvent() {
        	
            // System.out.println("DRIFT RESET IN MODEL #"+this.indexOriginal+" TO NEW BKG MODEL #"+this.bkgLearner.indexOriginal); 

	    		String [] eventLog = {String.valueOf(this.lastDriftOn), "DRIFT TO BKG MODEL", String.valueOf(this.indexOriginal), 
						String.valueOf(this.evaluator.getPerformanceMeasurements()[1].getValue()), 
						this.warningOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), 
						this.driftOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
						String.valueOf(this.createdOn), String.valueOf(this.evaluator.getFractionIncorrectlyClassified()), 
						String.valueOf(this.useRecurringLearner ? ConceptHistory.modelsOnWarning.size() : "N/A"), 
						String.valueOf(this.useRecurringLearner ? ConceptHistory.getNumberOfActiveWarnings() : "N/A"), 
						String.valueOf(this.useRecurringLearner ? ConceptHistory.modelsOnWarning : "N/A"), 
						"N/A", "N/A", "N/A"};    
				    		
        		return (new Event(eventLog));
	    }
        
        public Event getRecurringDriftEvent(Integer indexOfBestRankedInCH) {
	    		
	    		//// System.out.println(indexOfBestRankedInCH); // TODO: debugging
        	        	
            // System.out.println("RECURRING DRIFT RESET IN POSITION #"+this.indexOriginal+" TO MODEL #"+ConceptHistory.historyList.get(indexOfBestRankedInCH).ensembleIndex); //+this.bkgLearner.indexOriginal);   

        		String [] eventLog = {
    		        String.valueOf(this.lastDriftOn), "RECURRING DRIFT", String.valueOf(this.indexOriginal), 
		    		String.valueOf(this.evaluator.getPerformanceMeasurements()[1].getValue()), 
		    		this.warningOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), 
		    		this.driftOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), 
		    		String.valueOf(this.createdOn), String.valueOf(this.evaluator.getFractionIncorrectlyClassified()), 
		    		String.valueOf(this.useRecurringLearner ? ConceptHistory.modelsOnWarning.size() : "N/A"),
		    		String.valueOf(this.useRecurringLearner ? ConceptHistory.getNumberOfActiveWarnings() : "N/A"), 
		    		String.valueOf(this.useRecurringLearner ? ConceptHistory.modelsOnWarning : "N/A"), "N/A",
		    		String.valueOf(ConceptHistory.historyList.get(indexOfBestRankedInCH).ensembleIndex),
		    		String.valueOf(ConceptHistory.historyList.get(indexOfBestRankedInCH).createdOn)
        		};    
	    		
        		return (new Event(eventLog));
        }
                
        public Event getFalseAlarmEvent() {
        	
            // System.out.println("DRIFT RESET IN MODEL #"+this.indexOriginal+" TO NEW BKG MODEL #"+this.bkgLearner.indexOriginal); 

	    		String [] eventLog = {String.valueOf(this.lastDriftOn), "FALSE ALARM", String.valueOf(this.indexOriginal), 
						String.valueOf(this.evaluator.getPerformanceMeasurements()[1].getValue()), 
						this.warningOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), 
						this.driftOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
						String.valueOf(this.createdOn), String.valueOf(this.evaluator.getFractionIncorrectlyClassified()), 
						String.valueOf(this.useRecurringLearner ? ConceptHistory.modelsOnWarning.size() : "N/A"), 
						String.valueOf(this.useRecurringLearner ? ConceptHistory.getNumberOfActiveWarnings() : "N/A"), 
						String.valueOf(this.useRecurringLearner ? ConceptHistory.modelsOnWarning : "N/A"), 
						"N/A", "N/A", "N/A"};
				    		
        		return (new Event(eventLog));
	    }
        
		
    }
      
    /***
     * Inner class to assist with the multi-thread execution. 
     */
    protected class TrainingRunnable implements Runnable, Callable<Integer> {
        final private RCARFBaseLearner learner;
        final private Instance instance;
        final private double weight;
        final private long instancesSeen;

        public TrainingRunnable(RCARFBaseLearner learner, Instance instance, 
                double weight, long instancesSeen) {
            this.learner = learner;
            this.instance = instance;
            this.weight = weight;
            this.instancesSeen = instancesSeen;
        }

        @Override
        public void run() {
            learner.trainOnInstance(this.instance, this.weight, this.instancesSeen); // TODO. add the events log file here
        }

        @Override
        public Integer call() throws Exception {
            run();
            return 0;
        }
    }
    
    
    //Concept_history = (list of concept_representations)
    // Static and concurrent for all DTs that run in parallel
    public static class ConceptHistory {

    		// Concurrent Concept History List
    		public static ConcurrentHashMap<Integer,Concept> historyList; // = new ConcurrentHashMap<Integer,Concept> ();
    		public static int lastID = 0;
    		
    		// List of ensembles with an active warning used as to determine if the history list evaluators should be in use
    		public static ConcurrentHashMap<Integer,Boolean> modelsOnWarning; // = new ConcurrentHashMap<Integer,Boolean> ();

	    public static RCARFBaseLearner extractConcept(int key) {
	    		RCARFBaseLearner aux = historyList.get(key).getBaseLearner();
	    		historyList.remove(key);
			return aux;
	    }
	    
	    public static int getNumberOfActiveWarnings() {
	    		int count = 0;
	    		for(Boolean value: modelsOnWarning.values()) if (value) count++;
	    		return count;
	    }
	    
		public Set<Entry<Integer,Concept>> getConceptHistoryEntrySet() {
			return historyList.entrySet();
        }
	        
        // Getters
        public static int nextID() {
        		return lastID++;
        }
    }
    
    //Concept_representation = (model, last_weight, last_used_timestamp, conceptual_vector)
    public class Concept {

    		// Concept attributes
    		protected int ensembleIndex; // position that it had in the ensemble. for reference only.
    		protected int historyIndex; // id in concept history
    		
    		// Stats
    		protected long createdOn;
    		protected long instancesSeen;
    		protected double classifiedInstances;
    		protected double errorBeforeWarning;
    		
    		// Learner
    		public RCARFBaseLearner ConceptLearner;

    		// Constructor
    		public Concept (RCARFBaseLearner ConceptLearner, long createdOn, double classifiedInstances, long instancesSeen) {
    			// Extra info
	    		this.createdOn=createdOn;
	    		this.instancesSeen=instancesSeen;
	    		this.classifiedInstances=classifiedInstances;
	    		this.ensembleIndex=ConceptLearner.indexOriginal;

	    		// Learner
    			this.ConceptLearner = ConceptLearner;
    		}
    		
    		public void addHistoryID(int id){
    			this.historyIndex=id;
    		}
    		
	    	    
	    public int getEnsembleIndex() {
			return this.ensembleIndex;	    	
	    }
	    
	    public int getHistoryIndex() {
	    		return this.historyIndex;
	    }
	    
	    public RCARFBaseLearner getBaseLearner() {
			return this.ConceptLearner;	    	
	    }   
	    
	    // Setters
	    
	    public void setErrorBeforeWarning(double value) {
	    		this.errorBeforeWarning=value;
	    }
	}
    
    
    // Window-related parameters for classifier internal comparisons during the warning window 
    public class Window{
    	
	    	// Window properties
	    	int windowSize;   
	    	int windowDefaultSize;   
	    	int windowIncrements;
	    	int minWindowSize;
	    	double decisionThreshold;
	    	int windowResizePolicy;
	    	boolean backgroundDynamicWindowsFlag;
	    	boolean rememberWindowSize;	    	

		public Window(int windowSize, int windowIncrements, int minWindowSize, 
					  double decisionThreshold, boolean rememberWindowSize, 
					  boolean backgroundDynamicWindowsFlag, int windowResizePolicy) {
			this.windowSize=windowSize;
			this.windowDefaultSize=windowSize; // the default size of a window could change overtime if there is window sizw inheritance enabled
			this.windowIncrements=windowIncrements;
			this.minWindowSize=minWindowSize;
			this.decisionThreshold=decisionThreshold;
			this.backgroundDynamicWindowsFlag=backgroundDynamicWindowsFlag;
			this.windowResizePolicy=windowResizePolicy;
			this.rememberWindowSize=rememberWindowSize;
		}
		
		public Window copy() {
			return new Window(this.windowSize, this.windowIncrements, this.minWindowSize, 
					  this.decisionThreshold, this.rememberWindowSize, 
					  this.backgroundDynamicWindowsFlag, this.windowResizePolicy) ;
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
		
		public boolean getRememberSizeFlag(){
			return this.rememberWindowSize;
		}
		
		public void setRememberSizeFlag(boolean flag){
			this.rememberWindowSize=flag;
		}
		
		public int getResizingPolicy() {
			return this.windowResizePolicy;
		}

		public void setResizingPolicy(int value) {
			this.windowResizePolicy = value;
		}
		
		public boolean getDynamicWindowInOldModelsFlag() {
			return this.backgroundDynamicWindowsFlag;
		}

		public void getDynamicWindowInOldModelsFlag(boolean flag) {
			this.backgroundDynamicWindowsFlag = flag;
		}
		
    }

    // Object for events so the code is cleaner
    public class Event {
    	
    		// Fields for events log
		String instanceNumber;
		String event;
		String affectedPosition;
		String votingWeigth;
		String warningSetting;
		String driftSetting;
		String createdOn;
		String lastError;
		String numberOfModels;
		String numberOfActiveWarnings;
		String modelsOnWarning;
		String listOfApplicableConcepts;
		String recurringDriftToModelID;
		String driftToModelCreatedOn;
    	
		// Constructor from array
    		public Event (String [] eventDetails) {
    			 instanceNumber = eventDetails[0];
    			 event = eventDetails[1];
    			 affectedPosition = eventDetails[2];
    			 votingWeigth = eventDetails[3];
    			 warningSetting = eventDetails[4];
    			 driftSetting = eventDetails[5];
    			 createdOn= eventDetails[6];
    			 lastError = eventDetails[7];
    			 numberOfModels = eventDetails[8];
    			 numberOfActiveWarnings = eventDetails[9];
    			 modelsOnWarning = eventDetails[10];
    			 listOfApplicableConcepts = eventDetails[11];
    			 recurringDriftToModelID = eventDetails[12];
    			 driftToModelCreatedOn = eventDetails[13];
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

    		public String getNumberOfModels() {
    			return numberOfModels;
    		}

    		public void setNumberOfModels(String numberOfModels) {
    			this.numberOfModels = numberOfModels;
    		}

    		public String getNumberOfActiveWarnings() {
    			return numberOfActiveWarnings;
    		}

    		public void setNumberOfActiveWarnings(String numberOfActiveWarnings) {
    			this.numberOfActiveWarnings = numberOfActiveWarnings;
    		}

    		public String getModelsOnWarning() {
    			return modelsOnWarning;
    		}

    		public void setModelsOnWarning(String modelsOnWarning) {
    			this.modelsOnWarning = modelsOnWarning;
    		}

    		public String getListOfApplicableConcepts() {
    			return listOfApplicableConcepts;
    		}

    		public void setListOfApplicableConcepts(String listOfApplicableConcepts) {
    			this.listOfApplicableConcepts = listOfApplicableConcepts;
    		}

    		public String getRecurringDriftToModelID() {
    			return recurringDriftToModelID;
    		}

    		public void setRecurringDriftToModelID(String recurringDriftToModelID) {
    			this.recurringDriftToModelID = recurringDriftToModelID;
    		}

    		public String getDriftToModelCreatedOn() {
    			return driftToModelCreatedOn;
    		}

    		public void setDriftToModelCreatedOn(String driftToModelCreatedOn) {
    			this.driftToModelCreatedOn = driftToModelCreatedOn;
    		}
    	
    }
    
    // General auxiliar functions for logging events
    
    public Event getEventHeaders() {
    	
		String [] headers = {
				"instance_number", 
				"event_type", 
				"affected_position", //former 'model'
				"voting_weight", // voting weight for the three that presents an event. new 07/07/2018
				"warning_setting", 
				"drift_setting", 
				"affected_classifier_created_on", 
				"error_percentage", 
				"amount_of_models",
				"amount_of_active_warnings",
				"classifiers_on_warning",
				"applicable_concepts",
				"recurring_drift_to_history_id",
				"recurring_drift_to_classifier_created_on"
		};
	
		return (new Event(headers));

}
    
    /**
     * Method to register events such as Warning and Drifts in the event log file.
     * */
	public void logEvent(Event eventDetails) {		
		// Log processed instances, warnings and drifts in file of events
		//# instance, event, affected_position, affected_classifier_id last-error, #models;#active_warnings; models_on_warning, applicable_concepts_from_here, recurring_drift_to_history_id, drift_to_classifier_created_on
		eventsLogFile.println(
				String.join(";",
						eventDetails.getInstanceNumber(), 
						eventDetails.getEvent(), 
						eventDetails.getAffectedPosition(),
						eventDetails.getVotingWeigth(), // of the affected position (this is as represented in getVotesForInstance for the global ensemble), // new 07/07/2018
						eventDetails.getWarningSetting(), // WARNING SETTING of the affected position. new 07/07/2018
						eventDetails.getDriftSetting(), // DRIFT SETTING of the affected position. new 07/07/2018
						eventDetails.getCreatedOn(), // new, affected_classifier_was_created_on
        					eventDetails.getLastError(), 
        					eventDetails.getNumberOfModels(),
        					eventDetails.getNumberOfActiveWarnings(), // #active_warnings
        					eventDetails.getModelsOnWarning(), 
        					eventDetails.getListOfApplicableConcepts(), // applicable_concepts_from_here
        					eventDetails.getRecurringDriftToModelID(), // recurring_drift_to_history_id
        					eventDetails.getDriftToModelCreatedOn())
				);
		eventsLogFile.flush();
		
	}    
        
    
}
