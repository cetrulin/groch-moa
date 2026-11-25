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

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.Serializable;

/**
 * Optional semicolon-separated event logger for warnings, drifts, and Concept
 * History lifecycle events. Disabled unless an explicit output path is given.
 */
public class DriftEventLogger implements Serializable {

    private static final long serialVersionUID = 1L;

    public static final String[] HEADERS = {
        "instance_number", "event_type", "affected_position",
        "voting_weight",
        "warning_setting", "drift_setting", "affected_classifier_created_on", "error_percentage",
        "amount_of_classifiers", "amount_of_active_warnings", "classifiers_on_warning", "concept_history_snapshot",
        "recurring_drift_to_history_id", "recurring_drift_to_classifier_created_on",
        "gng_Seen_examples", "gng_prototypes", "topology_error"
    };

    private transient PrintWriter writer;

    public DriftEventLogger(String filePath, boolean disabled) {
        if (disabled || filePath == null || filePath.trim().isEmpty()) {
            this.writer = null;
            return;
        }
        try {
            this.writer = new PrintWriter(filePath);
            log(HEADERS);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            this.writer = null;
        }
    }

    public boolean isEnabled() {
        return this.writer != null;
    }

    public void log(String[] fields) {
        if (this.writer != null && fields != null) {
            this.writer.println(String.join(";", fields));
            this.writer.flush();
        }
    }
}
