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

import com.yahoo.labs.samoa.instances.Instances;
import java.io.Serializable;

/**
 * Strategy interface for computing similarity distances between a query window
 * and stored concept topology prototypes in {@link NA}.
 */
public interface ConceptDistance extends Serializable {

    /**
     * Computes the distance between two sets of instances/prototypes.
     */
    double distance(Instances query, Instances target);

    /**
     * Optional pre-extraction hook executed once per query window before
     * iterating over all candidate groups in Concept History.
     */
    default double[][] prepareQuery(Instances query) {
        return null;
    }

    /**
     * Fast-path distance evaluation using pre-extracted primitive matrices
     * when supported by the concrete strategy.
     */
    default double distance(Instances query, double[][] queryMatrix, Instances target, double[][] targetWeights) {
        return distance(query, target);
    }
}
