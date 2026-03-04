#pragma once

#include "relml/Column.h"
#include <vector>

namespace relml {

// Decomposes a TIMESTAMP column (unix seconds) into cyclical features.
// Output per row: [sin_month, cos_month, sin_day, cos_day, year_norm]
// 5 features total. No fitting required — pure deterministic transform.
class TimestampEncoder {
public:
    static constexpr std::size_t OUTPUT_DIM = 5;

    void fit(const Column& col);  // computes year mean/std for normalization

    // Returns flat matrix: num_rows x OUTPUT_DIM, row-major
    std::vector<float> transform(const Column& col) const;

private:
    float year_mean_ = 2000.f;
    float year_std_  = 20.f;
    bool  fitted_    = false;
};

} // namespace relml
