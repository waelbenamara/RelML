#pragma once

#include "relml/Column.h"
#include <vector>

namespace relml {

// Standardizes a NUMERICAL column to zero mean, unit variance.
// Null values are replaced by the column mean before standardization.
class NumericalEncoder {
public:
    void fit(const Column& col);
    std::vector<float> transform(const Column& col) const;

    float mean() const { return mean_; }
    float std()  const { return std_;  }

private:
    float mean_ = 0.f;
    float std_  = 1.f;
    bool  fitted_ = false;
};

} // namespace relml
