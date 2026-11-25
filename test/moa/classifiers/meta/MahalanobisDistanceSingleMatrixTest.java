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

import java.util.Random;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * JUnit bit-exact parity test for {@link MahalanobisDistanceSingleMatrix}.
 */
public class MahalanobisDistanceSingleMatrixTest extends TestCase {

    public MahalanobisDistanceSingleMatrixTest(String name) {
        super(name);
    }

    private static double legacyDistance(double[] v1, double[] v2, RealMatrix cov) {
        int d = v1.length;
        double det = Math.pow((new LUDecomposition(cov).getDeterminant()), 1 / d);
        double[] temp = new double[d];
        for (int i = 0; i < d; i++) {
            temp[i] = (v1[i] - v2[i]) * det;
        }
        RealMatrix m2 = new Array2DRowRealMatrix(new double[][] { temp });
        RealMatrix m3 = m2.multiply(new LUDecomposition(cov).getSolver().getInverse());
        RealMatrix m4 = m3.multiply((new Array2DRowRealMatrix(new double[][] { temp })).transpose());
        return Math.sqrt(m4.getEntry(0, 0));
    }

    public void testBitExactParityAgainstLegacy() {
        String prop = System.getProperty("moa.test.arffs", "");
        assertFalse("Set -Dmoa.test.arffs=path1;path2;...", prop.trim().isEmpty());
        String[] arffPaths = prop.split(";");

        MahalanobisDistanceSingleMatrix opt = new MahalanobisDistanceSingleMatrix(arffPaths);
        assertNotNull(opt.covarMatrix);
        RealMatrix cov = opt.covarMatrix;
        int d = cov.getRowDimension();
        Random rng = new Random(42);
        double[] scratch = new double[d];

        int pairs = 1000;
        for (int i = 0; i < pairs; i++) {
            double[] v1 = new double[d];
            double[] v2 = new double[d];
            for (int k = 0; k < d; k++) {
                v1[k] = rng.nextGaussian() * 5.0;
                v2[k] = rng.nextGaussian() * 5.0;
            }
            double ref = legacyDistance(v1, v2, cov);
            double fast = opt.distance(v1, v2, scratch);
            assertEquals("Bit mismatch at pair " + i,
                    Double.doubleToRawLongBits(ref),
                    Double.doubleToRawLongBits(fast));
        }
    }

    public static Test suite() {
        return new TestSuite(MahalanobisDistanceSingleMatrixTest.class);
    }

    public static void main(String[] args) {
        if (args.length > 0 && System.getProperty("moa.test.arffs") == null) {
            System.setProperty("moa.test.arffs", String.join(";", args));
        }
        junit.textui.TestRunner.run(suite());
    }
}
