/*
 *    GroCH.java
 *
 *    @author Andres L. Suarez-Cetrulo
 *
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
 *
 */
package moa.classifiers.meta;

/**
 * Compatibility alias extending {@link NA}.
 *
 * Allows experiment scripts invoking {@code meta.GroCH} to delegate directly
 * to the refactored {@link NA} implementation.
 */
public class GroCH extends NA {

    private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "GroCH (alias for NA) meta-classifier for recurring concepts using topology-based Concept History (Suarez-Cetrulo et al.).";
    }
}
