#include "relml/training/Adam.h"
#include <cmath>

namespace relml {

Adam::Adam(float lr, float beta1, float beta2, float eps)
    : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps)
{}

void Adam::step(std::vector<Parameter*>& params) {
    ++t_;
    float bc1 = 1.f - std::pow(beta1_, t_);
    float bc2 = 1.f - std::pow(beta2_, t_);

#pragma omp parallel for schedule(static)
    for (std::size_t idx = 0; idx < params.size(); ++idx) {
        Parameter* p = params[idx];
        for (std::size_t j = 0; j < p->size(); ++j) {
            float g = p->grad[j];
            p->m1[j] = beta1_ * p->m1[j] + (1.f - beta1_) * g;
            p->m2[j] = beta2_ * p->m2[j] + (1.f - beta2_) * g * g;
            float m_hat = p->m1[j] / bc1;
            float v_hat = p->m2[j] / bc2;
            p->data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
        p->zero_grad();  // clear gradients after update
    }
}

} // namespace relml