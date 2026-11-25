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
