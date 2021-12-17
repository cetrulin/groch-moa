/*
 * Change Detector Wrapper for NA.java
 */

package moa.classifiers.core.driftdetection;

import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;

public class GroCHChangeDetector extends AbstractChangeDetector {
	private static final long serialVersionUID = 2461734596321887617L;
		
	public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
			"Change detector for drifts and its parameters", ChangeDetector.class, "ADDM -a 1.0E-5 -p 1.0E-4");
	
	private ChangeDetector driftDetectionMethod;
	
	private boolean isWarningDetected;
	
    public GroCHChangeDetector() {
    	this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
    }
    
    public GroCHChangeDetector(ClassOption changeDetectorOption, ChangeDetector changeDetector) {
    	this.driftDetectionMethodOption = changeDetectorOption;
    	this.driftDetectionMethod = changeDetector;
    }
    
	@Override
	public void input(double inputValue) {
		boolean wasWarningZone = this.driftDetectionMethod.getWarningZone();
//		if (wasWarningZone) System.out.println("\n\nWARNING ZONE ON!!\n\n\n\n/n/n");
//		System.out.println("PREDICTION IN NA WRAPPER: "+inputValue);
		this.isWarningDetected = false;
		this.driftDetectionMethod.input(inputValue);
//		System.out.println("ADDM warning: "+((ADDM) this.driftDetectionMethod).adwinWarning.getEstimation());
//		System.out.println("ADDM drift: "+((ADDM) this.driftDetectionMethod).adwinDrift.getEstimation());

		if (this.driftDetectionMethodOption.getValueAsCLIString().contains("ADDM")) {
			this.isWarningDetected = ((ADDM) this.driftDetectionMethod).getWarning();
//			if (this.isWarningDetected) System.out.println("\n--\nstarted warning\n===================");
		} else {
	        if (!wasWarningZone && this.driftDetectionMethod.getWarningZone()) {
	            this.isWarningDetected = true;
	        } 
        }
	}
	
	@Override
	public void resetLearning() {
		this.driftDetectionMethod.resetLearning();
	}
	
	/*
	 * This resets learning for warning detector. 
	 * Only applicable for cases with two different detectors for drift and warning.
	 * */
	public void resetWarningDetector(){
		// Only relevant for ADDM
		if (this.driftDetectionMethodOption.getValueAsCLIString().contains("ADDM"))
			((ADDM) this.driftDetectionMethod).resetWarningDetector();
		// else: do nothing
	}
	
	/*
	 * This resets learning for drift (if there are many) or main change detector. 
	 * */
	public void resetDriftDetector(){ // Only relevant for ADDM
		((ADDM) this.driftDetectionMethod).resetDriftDetector();
	}
	
	public void setDriftDetector(ChangeDetector changeDetector) { // reset for all detectors!=ADDM
		this.driftDetectionMethod = changeDetector;
	}

	@Override
	public void getDescription(StringBuilder sb, int indent) {
		this.driftDetectionMethod.getDescription(sb, indent);
	}

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		this.driftDetectionMethod.prepareForUse();
	}
	
	@Override
	public boolean getWarningZone() {
		return this.driftDetectionMethod.getWarningZone();
	}
	
	/*
	 * Get new warnings (if a warning signal is started from this iteration)
	 * */
	public boolean getWarning() {
		return this.isWarningDetected;
	}
	
	@Override
    public boolean getChange() {
    	return this.driftDetectionMethod.getChange();
    }

	@Override
    public double getEstimation() {
		return this.driftDetectionMethod.getEstimation();
	}

	@Override
    public double getDelay() {
		return this.driftDetectionMethod.getDelay();
	}

	@Override
    public double[] getOutput() {
    	return this.driftDetectionMethod.getOutput();
    }

    @Override
    public ChangeDetector copy() {
    	return new GroCHChangeDetector(this.driftDetectionMethodOption, this.driftDetectionMethod.copy());
    }

}