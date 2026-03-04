#include "relml/training/TaskBuilder.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace relml {

TaskSplit build_rating_task(const Database& db, float threshold) {
    const Table& ratings = db.get_table("ratings");

    const Column& rating_c = ratings.get_column("rating");
    const Column& ts_c     = ratings.get_column("timestamp");

    struct Sample { int64_t row; int64_t ts; float label; };
    std::vector<Sample> samples;
    samples.reserve(ratings.num_rows());

    std::size_t n_pos = 0;
    for (std::size_t i = 0; i < ratings.num_rows(); ++i) {
        if (rating_c.is_null(i) || ts_c.is_null(i)) continue;
        float r     = static_cast<float>(rating_c.get_numerical(i));
        float label = (r >= threshold) ? 1.f : 0.f;
        if (label > 0.5f) ++n_pos;

        // timestamp is stored as a Unix integer — always read as numerical
        int64_t ts = static_cast<int64_t>(ts_c.get_numerical(i));
        samples.push_back({static_cast<int64_t>(i), ts, label});
    }

    std::sort(samples.begin(), samples.end(), [](const Sample& a, const Sample& b){
        return a.ts < b.ts;
    });

    std::size_t N     = samples.size();
    std::size_t t70   = N * 70 / 100;
    std::size_t t85   = N * 85 / 100;

    TaskSplit split;
    float train_pos = 0.f;

    for (std::size_t i = 0; i < N; ++i) {
        TaskSample s{samples[i].row, samples[i].label};
        if      (i < t70) { split.train.push_back(s); train_pos += s.label; }
        else if (i < t85)   split.val.push_back(s);
        else                split.test.push_back(s);
    }

    std::cout << "  Ratings: " << N
              << "  positive rate: " << std::fixed << std::setprecision(3)
              << (float)n_pos / N << "  (rating >= " << threshold << ")\n"
              << "  Samples — train: " << split.train.size()
              << "  val: "  << split.val.size()
              << "  test: " << split.test.size() << "\n"
              << "  Train label rate: " << train_pos / split.train.size() << "\n";

    return split;
}

void save_task_csv(const TaskSplit& task, const std::string& path,
                   const Database& db, const std::string& table,
                   const std::string& pk_col)
{
    const Table&  t   = db.get_table(table);
    const Column& pk  = t.get_column(pk_col);

    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("save_task_csv: cannot open " + path);
    f << pk_col << ",label,split\n";

    auto write = [&](const std::vector<TaskSample>& samples, const std::string& sp) {
        for (const auto& s : samples) {
            int64_t id = static_cast<int64_t>(pk.get_numerical(s.node_idx));
            f << id << "," << (int)s.label << "," << sp << "\n";
        }
    };
    write(task.train, "train");
    write(task.val,   "val");
    write(task.test,  "test");
    std::cout << "  Task saved to " << path << "\n";
}

} // namespace relml