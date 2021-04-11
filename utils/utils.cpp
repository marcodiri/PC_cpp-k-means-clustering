#include "utils.h"

bool compare(const float &value1, const float &value2, const int& precision) {
    auto magnitude = static_cast<int64_t>(std::pow(10, precision));
    auto intValue1 = static_cast<int64_t>(value1 * magnitude);
    auto intValue2 = static_cast<int64_t>(value2 * magnitude);
    return intValue1 == intValue2;
}