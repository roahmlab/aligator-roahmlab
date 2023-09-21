#pragma once

#include <type_traits>

namespace proxddp {

template <typename T> inline T intExp2(T N) {
  static_assert(std::is_integral<T>::value, "Integral type required");
  return 1 << N;
}

template <typename T> inline T intLog2(T N) {
  static_assert(std::is_integral<T>::value, "Integral type required");
  T shift = 0;
  while (((N >> shift) & 1) != 1) {
    shift++;
  }
  return shift;
}

} // namespace proxddp
