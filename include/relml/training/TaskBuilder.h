#pragma once

#include "relml/Database.h"
#include "relml/training/TaskSpec.h"  // TaskSample, TaskSplit live here now

#include <string>

namespace relml {

// Convenience builder for the MovieLens rating prediction task.
// Equivalent to constructing a TaskSpec with:
//   target_table  = "ratings",  target_column = "rating"
//   task_type     = BinaryClassification
//   label_transform: Threshold(threshold, inclusive=true)
//   split_strategy: Temporal (uses "timestamp" column)
//
// Use TaskSpec::build_split() directly for any other database or task.
TaskSplit build_rating_task(const Database& db, float threshold = 4.f);

// Persist a TaskSplit to CSV for inspection or offline use.
void save_task_csv(const TaskSplit& task, const std::string& path,
                   const Database& db, const std::string& table,
                   const std::string& pk_col);

} // namespace relml