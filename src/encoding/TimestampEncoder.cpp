#include "relml/encoding/TimestampEncoder.h"
#include <cmath>
#include <ctime>

namespace relml {

static constexpr float PI = 3.14159265358979f;

static std::tm to_tm(int64_t unix_sec) {
    std::time_t t = static_cast<std::time_t>(unix_sec);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    return tm;
}

void TimestampEncoder::fit(const Column& col) {
    double sum = 0.0;
    double sum_sq = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < col.size(); ++i) {
        if (col.is_null(i)) continue;
        std::tm tm = to_tm(col.get_timestamp(i));
        float year = static_cast<float>(tm.tm_year + 1900);
        sum    += year;
        sum_sq += year * year;
        ++count;
    }
    if (count == 0) { fitted_ = true; return; }
    year_mean_ = static_cast<float>(sum / count);
    float var  = static_cast<float>(sum_sq / count) - year_mean_ * year_mean_;
    year_std_  = (var > 1e-8f) ? std::sqrt(var) : 1.f;
    fitted_    = true;
}

std::vector<float> TimestampEncoder::transform(const Column& col) const {
    std::size_t N = col.size();
    std::vector<float> out(N * OUTPUT_DIM, 0.f);

    for (std::size_t i = 0; i < N; ++i) {
        float* row = out.data() + i * OUTPUT_DIM;
        if (col.is_null(i)) continue;  // leave as zeros

        std::tm tm = to_tm(col.get_timestamp(i));
        int month  = tm.tm_mon + 1;     // 1-12
        int day    = tm.tm_mday;        // 1-31
        float year = static_cast<float>(tm.tm_year + 1900);

        row[0] = std::sin(2.f * PI * month / 12.f);
        row[1] = std::cos(2.f * PI * month / 12.f);
        row[2] = std::sin(2.f * PI * day   / 31.f);
        row[3] = std::cos(2.f * PI * day   / 31.f);
        row[4] = (year - year_mean_) / year_std_;
    }
    return out;
}

} // namespace relml
