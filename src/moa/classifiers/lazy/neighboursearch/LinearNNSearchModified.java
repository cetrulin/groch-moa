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

package moa.classifiers.lazy.neighboursearch;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

/**
 * Extension of MOA's {@link LinearNNSearch} that supports nearest-neighbour
 * lookup including the class attribute via {@link EuclideanDistanceModified}.
 */
public class LinearNNSearchModified extends LinearNNSearch {

    private static final long serialVersionUID = 1915484723703917241L;

    public LinearNNSearchModified() {
        super();
        this.m_DistanceFunction = new EuclideanDistanceModified();
    }

    public LinearNNSearchModified(Instances insts) {
        super(insts);
        this.m_DistanceFunction = new EuclideanDistanceModified(insts);
    }

    public Instances kNearestNeighboursUsingClass(Instance target, int kNN) throws Exception {
        MyHeap heap = new MyHeap(kNN);
        EuclideanDistanceModified distFn = (EuclideanDistanceModified) m_DistanceFunction;
        int firstkNN = 0;

        for (int i = 0; i < m_Instances.numInstances(); i++) {
            if (target == m_Instances.instance(i)) {
                continue;
            }
            if (firstkNN < kNN) {
                double distance = distFn.distanceWithClass(target, m_Instances.instance(i), Double.POSITIVE_INFINITY, true);
                if (distance == 0.0 && m_SkipIdentical) {
                    if (i < m_Instances.numInstances() - 1) {
                        continue;
                    } else {
                        heap.put(i, distance);
                    }
                }
                heap.put(i, distance);
                firstkNN++;
            } else {
                MyHeapElement temp = heap.peek();
                double distance = distFn.distanceWithClass(target, m_Instances.instance(i), temp.distance, true);
                if (distance == 0.0 && m_SkipIdentical) {
                    continue;
                }
                if (distance < temp.distance) {
                    heap.putBySubstitute(i, distance);
                } else if (distance == temp.distance) {
                    heap.putKthNearest(i, distance);
                }
            }
        }

        int total = heap.size() + heap.noOfKthNearest();
        Instances neighbours = new Instances(m_Instances, total);
        m_Distances = new double[total];
        int[] indices = new int[total];
        int pos = 1;
        while (heap.noOfKthNearest() > 0) {
            MyHeapElement h = heap.getKthNearest();
            indices[total - pos] = h.index;
            m_Distances[total - pos] = h.distance;
            pos++;
        }
        while (heap.size() > 0) {
            MyHeapElement h = heap.get();
            indices[total - pos] = h.index;
            m_Distances[total - pos] = h.distance;
            pos++;
        }

        m_DistanceFunction.postProcessDistances(m_Distances);
        for (int k = 0; k < indices.length; k++) {
            neighbours.add(m_Instances.instance(indices[k]));
        }
        return neighbours;
    }
}
