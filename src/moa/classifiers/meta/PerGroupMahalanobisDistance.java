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
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import weka.core.MahalanobisDistance;

/**
 * Per-group Mahalanobis distance strategy (when a single global covariance
 * matrix is not used).
 */
public class PerGroupMahalanobisDistance implements ConceptDistance {

    private static final long serialVersionUID = 1L;

    @Override
    public double distance(Instances query, Instances target) {
        SamoaToWekaInstanceConverter converter = new SamoaToWekaInstanceConverter();
        MahalanobisDistance m = new MahalanobisDistance();
        return m.distance(converter.wekaInstances(query), converter.wekaInstances(target));
    }
}
