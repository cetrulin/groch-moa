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

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.lazy.neighboursearch.EuclideanDistanceModified;
import moa.classifiers.lazy.neighboursearch.LinearNNSearchModified;

/**
 * Euclidean nearest-neighbour concept distance strategy including the class
 * attribute via {@link EuclideanDistanceModified} and {@link LinearNNSearchModified}.
 */
public class EuclideanConceptDistance implements ConceptDistance {

    private static final long serialVersionUID = 1L;
    private final boolean debugNN;

    public EuclideanConceptDistance(boolean debugNN) {
        this.debugNN = debugNN;
    }

    @Override
    public double distance(Instances query, Instances target) {
        int nPrototypes = 1;
        try {
            EuclideanDistanceModified d = new EuclideanDistanceModified(query);
            double[] dist = new double[query.numInstances()];
            double totalDist = 0.0;
            for (int i = 0; i < query.numInstances(); i++) {
                Instances nearest = getNearestPrototypes(query.get(i), target, nPrototypes);
                for (int j = 0; j < nearest.numInstances(); j++) {
                    dist[i] += d.distanceUsingClass(query.get(i), nearest.get(j));
                }
                dist[i] /= nearest.numInstances();
                if (this.debugNN) {
                    assert nearest.numInstances() == nPrototypes;
                }
            }
            for (int i = 0; i < dist.length; i++) {
                totalDist += dist[i];
            }
            return totalDist / query.numInstances();
        } catch (Exception e) {
            e.printStackTrace();
            return Double.NaN;
        }
    }

    private Instances getNearestPrototypes(Instance inst, Instances topologyPrototypes, int nPrototypes) throws Exception {
        LinearNNSearchModified nnSearch = new LinearNNSearchModified();
        nnSearch.setInstances(topologyPrototypes);
        return nnSearch.kNearestNeighboursUsingClass(inst, nPrototypes);
    }
}
