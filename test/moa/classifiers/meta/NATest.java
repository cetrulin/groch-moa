/*
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
 */

package moa.classifiers.meta;

import com.yahoo.labs.samoa.instances.Instance;
import java.io.File;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import moa.streams.generators.SEAGenerator;

/**
 * JUnit integration test suite for {@link NA} covering default event-log
 * suppression, Euclidean and Mahalanobis strategies, and RDDM/ADDM detectors.
 */
public class NATest extends TestCase {

    public NATest(String name) {
        super(name);
    }

    private void runStreamSmoke(NA learner, int numInstances) {
        SEAGenerator stream = new SEAGenerator();
        stream.prepareForUse();

        learner.setModelContext(stream.getHeader());
        learner.prepareForUse();

        for (int i = 0; i < numInstances; i++) {
            Instance inst = stream.nextInstance().getData();
            double[] votes = learner.getVotesForInstance(inst);
            assertNotNull("Votes must not be null at step " + i, votes);
            learner.trainOnInstance(inst);
        }
        assertTrue("measureByteSize must not throw", learner.measureByteSize() >= -1);
    }

    public void testDefaultEventLogDisabledAndEuclideanRDDM() {
        File defaultLog = new File("./NA_events_log.txt");
        if (defaultLog.exists()) {
            defaultLog.delete();
        }
        NA learner = new NA();
        learner.baseLearnerOption.setValueViaCLIString("trees.HoeffdingTree");
        learner.driftDetectionMethodOption.setValueViaCLIString("RDDM -o 2.0");
        learner.distanceMetricOption.setChosenLabel("Euclidean");
        learner.consoleLogLevelOption.setValue(0);
        runStreamSmoke(learner, 1500);
        assertFalse("Event log must be disabled by default when -e is empty", defaultLog.exists());
    }

    public void testADDMDetectorRoute() {
        NA learner = new NA();
        learner.baseLearnerOption.setValueViaCLIString("trees.HoeffdingTree");
        learner.driftDetectionMethodOption.setValueViaCLIString("ADDM -a 1.0E-5 -p 1.0E-4");
        learner.distanceMetricOption.setChosenLabel("Euclidean");
        learner.consoleLogLevelOption.setValue(0);
        runStreamSmoke(learner, 1500);
    }

    public void testPerGroupMahalanobisRoute() {
        NA learner = new NA();
        learner.baseLearnerOption.setValueViaCLIString("trees.HoeffdingTree");
        learner.driftDetectionMethodOption.setValueViaCLIString("RDDM -o 2.0");
        learner.distanceMetricOption.setChosenLabel("Mahalanobis");
        learner.numberOfMatricesOption.setChosenIndex(0);
        learner.consoleLogLevelOption.setValue(0);
        runStreamSmoke(learner, 1500);
    }

    public static Test suite() {
        return new TestSuite(NATest.class);
    }

    public static void main(String[] args) {
        junit.textui.TestRunner.run(suite());
    }
}
