/*
 *    ADWINChangeDetector.java
 *    Copyright (C) 2008 University of Waikato, Hamilton, New Zealand
 *    @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
package moa.classifiers.core.driftdetection;

import com.github.javacliparser.FloatOption;
import moa.core.ObjectRepository;
import moa.tasks.TaskMonitor;

/**
 * Drift detection method based in ADWIN. ADaptive sliding WINdow is a change
 * detector and estimator. It keeps a variable-length window of recently seen
 * items, with the property that the window has the maximal length statistically
 * consistent with the hypothesis "there has been no change in the average value
 * inside the window".
 *
 *
 * @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 * @version $Revision: 7 $
 */
public class EvolvingADWINChangeDetector extends AbstractEvolvingChangeDetector {

    protected EvolvingADWIN adwin;

    public FloatOption deltaAdwinOption = new FloatOption("deltaAdwin", 'a',
            "Delta of Adwin change detection", 0.002, 0.0, 1.0);
    
    public EvolvingADWINChangeDetector (double deltaAdwin) {
    		// debug
    		//System.out.println();
    		//System.out.println("Inicializando nuevo ADWIN con valor "+deltaAdwin);
    		//System.out.println("El option tenia valor "+deltaAdwinOption.getValue());
    		deltaAdwinOption.setValue(deltaAdwin);
    		//System.out.println("Ahora tiene valor "+deltaAdwinOption.getValue());
    		//System.out.println();
    		// resetLearning(); // may not be needed here as it's called in 'input'
    }
    
    @Override
    public void input(double inputValue) {
    		// debug
    		//System.out.println();
		//System.out.println("Ahora tiene valor "+deltaAdwinOption.getValue());
		//System.out.println();
        if (this.adwin == null) {
            resetLearning();
        }
		//System.out.println("En adwin vale "+adwin.getDelta());

        double ErrEstim = this.adwin.getEstimation();
        if(adwin.setInput(inputValue)) {
            if (this.adwin.getEstimation() > ErrEstim) {
                this.isChangeDetected = true;
            }
        }
        this.isWarningZone = false;
        this.delay = 0.0;
        this.estimation = adwin.getEstimation();
    }
    
    @Override
    public void resetLearning() {
        adwin = new EvolvingADWIN((double) this.deltaAdwinOption.getValue());
    }
    
    /* added by asuarez for RCARF 22/05/2018 */
    @Override
    public void setDelta (double d) {
    		adwin.setDelta(d);
    }

    @Override
    public double getDelta() {
    		return adwin.getDelta();
    }
    /* ***************************************/
    
    @Override
    public void getDescription(StringBuilder sb, int indent) {
        // TODO Auto-generated method stub
    }

    @Override
    protected void prepareForUseImpl(TaskMonitor monitor,
            ObjectRepository repository) {
        // TODO Auto-generated method stub
    }
}
