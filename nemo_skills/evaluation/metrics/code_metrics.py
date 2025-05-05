# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics


class CodeMetrics(BaseMetrics):
    def __init__(self):
        self.reset()

    def update_comb_metric(self, perf_dict, current_correct_base, current_correct_plus):
        """Helper to update counts for non-averaged metrics."""
        perf_dict["total_correct"] += int(current_correct_base)
        perf_dict["total_correct_plus"] += int(current_correct_plus)

    def update_comb_metric_averaged(self, perf_dict, current_correct_base_avg, current_correct_plus_avg):
        """Helper to update counts for averaged metrics."""
        perf_dict["total_correct"] += current_correct_base_avg
        perf_dict["total_correct_plus"] += current_correct_plus_avg

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        self.total += 1

        k = len(predictions)

        if k == 1:
            # Single decoding
            current_correct_base = predictions[0]['is_correct']
            current_correct_plus = predictions[0]['is_correct-plus']
            self.update_comb_metric(
                self.agg_mode_dict["greedy"], current_correct_base, current_correct_plus
            )
        else:
            # Multiple decodings - iterate through k values
            for current_k in range(k, 0, -1):
                current_predictions = predictions[:current_k]

                # Pass@K
                pass_k_correct = any([elem['is_correct'] for elem in current_predictions])
                pass_k_correct_plus = any([elem['is_correct-plus'] for elem in current_predictions])
                self.update_comb_metric(
                    self.agg_mode_dict[f"pass@{current_k}"], pass_k_correct, pass_k_correct_plus
                )

                # Pass@1[K] - mean of pass@1 across the top k generations
                pass_1_k_correct_avg = sum([elem['is_correct'] for elem in current_predictions]) / current_k
                pass_1_k_correct_plus_avg = sum([elem['is_correct-plus'] for elem in current_predictions]) / current_k
                self.update_comb_metric_averaged(
                    self.agg_mode_dict[f"pass@1[{current_k}]"], pass_1_k_correct_avg, pass_1_k_correct_plus_avg
                )

    def get_metrics(self):
        metrics_dict = {}
        for agg_mode, agg_metric_dict in self.agg_mode_dict.items():
            metrics_dict[agg_mode] = {
                "num_entries": self.total,
                "passing_base_tests": agg_metric_dict["total_correct"] / self.total * 100.0,
                "passing_plus_tests": agg_metric_dict["total_correct_plus"] / self.total * 100.0,
            }

        return metrics_dict

    def reset(self):
        self.total = 0
        # Store metrics per aggregation mode
        self.agg_mode_dict = defaultdict(lambda: defaultdict(float))

    def max_aggregations_to_print(self):
        """Limit printing to only the metrics for the highest k."""
        # Corresponds to pass@k and pass@1[k] for the highest k
        # plus the greedy result if available (implicitly handled by summary script)
        return 2
