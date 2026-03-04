#pragma once

#include "relml/Parameter.h"
#include <vector>

namespace relml {

// Adam optimizer.
// Uses m1/m2 stored directly in each Parameter — no separate state maps.
// Zeros gradients after each step so the next backward starts clean.
class Adam {
public:
    explicit Adam(float lr = 1e-3f, float beta1 = 0.9f,
                  float beta2 = 0.999f, float eps = 1e-8f);

    // Updates all parameters, then zeros their gradients.
    void step(std::vector<Parameter*>& params);

private:
    float lr_, beta1_, beta2_, eps_;
    int   t_ = 0;
};

} // namespace relml