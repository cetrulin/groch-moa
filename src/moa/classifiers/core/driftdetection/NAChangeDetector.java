/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package moa.classifiers.core.driftdetection;

import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;

/**
 * Adapter around MOA {@link ChangeDetector} implementations that normalizes
 * warning-edge detection and isolates dual-detector behavior (e.g., {@link ADDM})
 * from meta-classifiers such as {@code NA}.
 */
public class NAChangeDetector extends AbstractChangeDetector {

    private static final long serialVersionUID = 2461734596321887617L;

    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
            "Change detector for drifts and its parameters", ChangeDetector.class, "ADDM -a 1.0E-5 -p 1.0E-4");

    private ChangeDetector driftDetectionMethod;
    private boolean isWarningDetected;

    public NAChangeDetector() {
        this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
    }

    public NAChangeDetector(ClassOption changeDetectorOption, ChangeDetector changeDetector) {
        this.driftDetectionMethodOption = changeDetectorOption;
        this.driftDetectionMethod = changeDetector;
    }

    /**
     * Returns {@code true} when the wrapped detector maintains independent
     * warning and drift sub-detectors (currently {@link ADDM}).
     */
    public boolean isDualDetector() {
        return this.driftDetectionMethod instanceof ADDM;
    }

    @Override
    public void input(double inputValue) {
        boolean wasWarningZone = this.driftDetectionMethod.getWarningZone();
        this.isWarningDetected = false;
        this.driftDetectionMethod.input(inputValue);

        if (isDualDetector()) {
            this.isWarningDetected = ((ADDM) this.driftDetectionMethod).getWarning();
        } else if (!wasWarningZone && this.driftDetectionMethod.getWarningZone()) {
            this.isWarningDetected = true;
        }
    }

    @Override
    public void resetLearning() {
        this.driftDetectionMethod.resetLearning();
    }

    public void resetWarningDetector() {
        if (isDualDetector()) {
            ((ADDM) this.driftDetectionMethod).resetWarningDetector();
        }
    }

    public void resetDriftDetector() {
        if (isDualDetector()) {
            ((ADDM) this.driftDetectionMethod).resetDriftDetector();
        }
    }

    public void resetAfterDrift(ChangeDetector freshDetector) {
        if (isDualDetector()) {
            ((ADDM) this.driftDetectionMethod).resetDriftDetector();
        } else {
            this.driftDetectionMethod = freshDetector;
        }
    }

    public void setDriftDetector(ChangeDetector changeDetector) {
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
        return new NAChangeDetector(this.driftDetectionMethodOption, this.driftDetectionMethod.copy());
    }
}
