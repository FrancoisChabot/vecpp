//  Copyright 2018 Francois Chabot
//  (francois.chabot.dev@gmail.com)
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
#ifndef VECPP_SINGLE_INCLUDE_H_
#define VECPP_SINGLE_INCLUDE_H_
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#ifdef _MSVC_LANG
#if _MSVC_LANG < 201703L
#error C++17 support is required
#endif
#elif __cplusplus < 201703L
#error C++17 support is required
#endif
#define VECPP_VERSION_MAJOR 0
#define VECPP_VERSION_MINOR 0
#define VECPP_VERSION_PATCH 1
#define VECPP_VERSION 001
#ifndef VECPP_NAMESPACE
#define VECPP_NAMESPACE vecpp
#endif

namespace VECPP_NAMESPACE {
using Flags = int;
namespace flags {
constexpr int compile_time = 1;
constexpr int testing = 0x80000000;
}  // namespace flags
constexpr bool is_ct(Flags f) { return f && flags::compile_time != 0; }
}  // namespace VECPP_NAMESPACE

namespace VECPP_NAMESPACE {
template <typename Scalar>
constexpr Scalar pi = Scalar(3.1415926535897932385);
template <typename Scalar>
constexpr Scalar half_pi = pi<Scalar> / Scalar(2);
template <typename Scalar>
constexpr Scalar quarter_pi = pi<Scalar> / Scalar(4);
template <typename Scalar>
constexpr Scalar two_pi = pi<Scalar>* Scalar(2);
}  // namespace VECPP_NAMESPACE

namespace VECPP_NAMESPACE {
namespace non_cste {
template <typename T>
T sqrt(const T& v) {
  return std::sqrt(v);
}
template <typename T>
T pow(const T& x, const T& n) {
  return std::pow(x, n);
}
template <typename T>
T exp(const T& v) {
  return std::exp(v);
}
template <typename T>
T ceil(const T& v) {
  return std::ceil(v);
}
template <typename T>
T floor(const T& v) {
  return std::floor(v);
}
template <typename T>
T fract(const T& v) {
  return v - floor(v);
}
template <typename T>
T round(const T& v) {
  return std::round(v);
}
template <typename T>
T trunc(const T& v) {
  return std::trunc(v);
}
template <typename T>
T mod(const T& v, const T& d) {
  if constexpr (std::is_integral_v<T>) {
    return v % d;
  } else {
    return std::fmod(v, d);
  }
}
}  // namespace non_cste
namespace cste {
constexpr unsigned long long factorial(std::size_t N) {
  unsigned long long result = 1;
  for(unsigned long long i = 1 ; i <= N ; ++i) {
    result *= i;
  }
  return result;
}
template <typename T>
constexpr T sqrt(const T& v) {
  if (v == T(0)) {
    return v;
  }
  T r = v;
  // A lazy newton-rhapson for now.
  while (1) {
    T tmp = T(0.5) * (r + v / r);
    if (tmp == r) {
      break;
    }
    r = tmp;
  }
  return r;
}
template <typename T>
constexpr T pow(const T& x, const T& n) {
  assert(false);
}
template <typename T>
constexpr T exp(const T& v) {
  assert(false);
}
template <typename T>
constexpr T ceil(const T& v) {
  long long int x = static_cast<long long int>(v);
  if (v == T(x) || v < T(0)) {
    return T(x);
  }
  return T(x + 1);
}
template <typename T>
constexpr T floor(const T& v) {
  long long int x = static_cast<long long int>(v);
  if (v == T(x) || v > T(0)) {
    return T(x);
  }
  return T(x - 1);
}
template <typename T>
constexpr T round(const T& v) {
  return floor(v + T(0.5));
}
template <typename T>
constexpr T trunc(const T& v) {
  long long int x = static_cast<long long int>(v);
  return T(x);
}
template <typename T>
constexpr T fract(const T& v) {
  return v - floor(v);
}
template <typename T>
constexpr T mod(const T& v, const T& d) {
  if constexpr (std::is_integral_v<T>) {
    return v % d;
  } else {
    return v - floor(v / d) * d;
  }
}
}  // namespace cste
template <Flags f = 0, typename ScalarT>
constexpr ScalarT abs(const ScalarT& v) {
  return v < ScalarT(0) ? -v : v;
}
template <Flags f = 0, typename ScalarT>
constexpr ScalarT ceil(const ScalarT& v) {
  if constexpr (!is_ct(f)) {
    return non_cste::ceil(v);
  } else {
    return cste::ceil(v);
  }
}
template <Flags f = 0, typename ScalarT>
constexpr ScalarT exp(const ScalarT& v) {
  if constexpr (!is_ct(f)) {
    return non_cste::exp(v);
  } else {
    return cste::exp(v);
  }
}
template <Flags f = 0, typename ScalarT>
constexpr ScalarT floor(const ScalarT& v) {
  if constexpr (!is_ct(f)) {
    return non_cste::floor(v);
  } else {
    return cste::floor(v);
  }
}
template <Flags f = 0, typename ScalarT>
constexpr ScalarT round(const ScalarT& v) {
  if constexpr (!is_ct(f)) {
    return non_cste::round(v);
  } else {
    return cste::round(v);
  }
}
template <Flags f = 0, typename ScalarT>
constexpr ScalarT sign(const ScalarT& v) {
  return v >= 0.0f ? 1.0f : -1.0f;
}
template <Flags f = 0, typename ScalarT>
constexpr ScalarT trunc(const ScalarT& v) {
  if constexpr (!is_ct(f)) {
    return non_cste::trunc(v);
  } else {
    return cste::trunc(v);
  }
}
template <Flags f = 0, typename ScalarT>
constexpr ScalarT mod(const ScalarT& v, const ScalarT& d) {
  if constexpr (!is_ct(f)) {
    return non_cste::mod(v, d);
  } else {
    return cste::mod(v, d);
  }
}
template <Flags f = 0, typename ScalarT>
constexpr ScalarT fract(const ScalarT& v) {
  if constexpr (!is_ct(f)) {
    return non_cste::fract(v);
  } else {
    return cste::fract(v);
  }
}
template <Flags f = 0, typename ScalarT>
constexpr ScalarT step(const ScalarT& edge, const ScalarT& x) {
  return x < edge ? 0.0f : 1.0f;
}
template <Flags f = 0, typename ScalarT>
constexpr ScalarT min(const ScalarT& lhs, const ScalarT& rhs) {
  return std::min(lhs, rhs);
}
template <Flags f = 0, typename ScalarT>
constexpr ScalarT max(const ScalarT& lhs, const ScalarT& rhs) {
  return std::max(lhs, rhs);
}
template <Flags f = 0, typename ScalarT>
constexpr ScalarT clamp(const ScalarT& v, const ScalarT& low,
                        const ScalarT& high) {
  return std::clamp(v, low, high);
}
template <Flags f = 0, typename ScalarT, typename PctT>
constexpr ScalarT lerp(const ScalarT& from, const ScalarT& to,
                       const PctT& pct) {
  return from + (to - from) * pct;
}
template <Flags f = 0, typename ScalarT>
constexpr ScalarT pow(const ScalarT& x, const ScalarT& n) {
  if constexpr (!is_ct(f)) {
    return non_cste::pow(x, n);
  } else {
    return cste::pow(x, n);
  }
}
template <Flags f = 0, typename T>
constexpr T sqrt(const T& v) {
  if constexpr (!is_ct(f)) {
    return non_cste::sqrt(v);
  } else {
    return cste::sqrt(v);
  }
}
}  // namespace VECPP_NAMESPACE

namespace VECPP_NAMESPACE {
template <typename T, Flags f = 0>
class Angle {
 public:
  using value_type = T;
  static constexpr Flags flags = f;
  static constexpr Angle from_rad(const value_type&);
  static constexpr Angle from_deg(const value_type&);
  // The argument MUST be in the ]-PI, PI] range.
  static constexpr Angle from_clamped_rad(const value_type&);
  // The argument MUST be in the ]-180, 180] range.
  static constexpr Angle from_clamped_deg(const value_type&);
  constexpr value_type as_deg() const;
  constexpr value_type as_rad() const;
  constexpr const value_type& raw() const;
  template <int new_flags>
  constexpr operator Angle<T, new_flags>() const;
 private:
  value_type value_;
  // Constructs an angle from a constrained radian value.
  explicit constexpr Angle(const T&);
};
template <typename T, Flags f>
constexpr Angle<T, f | flags::compile_time> ct(const Angle<T, f>& v) {
  return v;
}
template <typename T, Flags f>
template <int new_flags>
constexpr Angle<T, f>::operator Angle<T, new_flags>() const {
  return Angle<T, new_flags>::from_clamped_rad(value_);
}
template <typename T, Flags f>
constexpr Angle<T, f> operator-(const Angle<T, f>& rhs) {
  T value = rhs.as_rad();
  // Special case, we keep positive pi.
  if (value != pi<T>) {
    value = -value;
  }
  return Angle<T, f>::from_clamped_rad(value);
}
template <typename T, Flags f>
constexpr Angle<T, f>& operator+=(Angle<T, f>& lhs, const Angle<T, f>& rhs) {
  T val = lhs.as_rad() + rhs.as_rad();
  // Since both lhs and rhs are in the ]-PI,PI] range, the sum is in the
  // ]-2PI-1,2PI] range, so we can make assumptions in the constraining process.
  if (val > pi<T>) {
    val -= two_pi<T>;
  } else if (val <= -pi<T>) {
    val += two_pi<T>;
  }
  lhs = Angle<T, f>::from_clamped_rad(val);
  return lhs;
}
template <typename T, Flags f>
constexpr Angle<T, f> operator+(const Angle<T, f>& lhs,
                                const Angle<T, f>& rhs) {
  auto result = lhs;
  result += rhs;
  return result;
}
template <typename T, Flags f>
constexpr Angle<T, f>& operator-=(Angle<T, f>& lhs, const Angle<T, f>& rhs) {
  T val = lhs.as_rad() - rhs.as_rad();
  // Since both lhs and rhs are in the ]-PI,PI] range, the difference is in the
  // ]-2PI,2PI[ range, so we can make assumptions in the constraining process.
  if (val > pi<T>) {
    val -= two_pi<T>;
  } else if (val <= -pi<T>) {
    val += two_pi<T>;
  }
  lhs = Angle<T, f>::from_clamped_rad(val);
  return lhs;
}
template <typename T, Flags f>
constexpr Angle<T, f> operator-(const Angle<T, f>& lhs,
                                const Angle<T, f>& rhs) {
  auto result = lhs;
  result -= rhs;
  return result;
}
template <typename T, Flags f>
constexpr Angle<T, f>& operator*=(Angle<T, f>& lhs, const T& rhs) {
  lhs = Angle<T, f>::from_rad(lhs.as_rad() * rhs);
  return lhs;
}
template <typename T, Flags f>
constexpr Angle<T, f> operator*(const Angle<T, f>& lhs, const T& rhs) {
  auto result = lhs;
  result *= rhs;
  return result;
}
template <typename T, Flags f>
constexpr Angle<T, f> operator*(const T& lhs, const Angle<T, f>& rhs) {
  return rhs * lhs;
}
template <typename T, Flags f>
constexpr Angle<T, f>& operator/=(Angle<T, f>& lhs, const T& rhs) {
  lhs = Angle<T, f>::from_rad(lhs.as_rad() / rhs);
  return lhs;
}
template <typename T, Flags f>
constexpr Angle<T, f> operator/(const Angle<T, f>& lhs, const T& rhs) {
  auto result = lhs;
  result /= rhs;
  return result;
}
template <typename T, Flags f1, Flags f2>
constexpr bool operator==(const Angle<T, f1>& lhs, const Angle<T, f2>& rhs) {
  return lhs.raw() == rhs.raw();
}
template <typename T, Flags f1, Flags f2>
constexpr bool operator!=(const Angle<T, f1>& lhs, const Angle<T, f2>& rhs) {
  return lhs.raw() != rhs.raw();
}
template <typename T, Flags f1, Flags f2>
constexpr bool operator<(const Angle<T, f1>& lhs, const Angle<T, f2>& rhs) {
  return lhs.raw() < rhs.raw();
}
template <typename T, Flags f1, Flags f2>
constexpr bool operator>(const Angle<T, f1>& lhs, const Angle<T, f2>& rhs) {
  return lhs.raw() > rhs.raw();
}
template <typename T, Flags f1, Flags f2>
constexpr bool operator<=(const Angle<T, f1>& lhs, const Angle<T, f2>& rhs) {
  return lhs.raw() <= rhs.raw();
}
template <typename T, Flags f1, Flags f2>
constexpr bool operator>=(const Angle<T, f1>& lhs, const Angle<T, f2>& rhs) {
  return lhs.raw() >= rhs.raw();
}
template <typename T, Flags f>
std::ostream& operator<<(std::ostream& stream, const Angle<T, f>& v) {
  return stream << v.as_deg() << "°";
}
template <typename T, Flags f>
constexpr Angle<T, f>::Angle(const T& v) : value_(v) {}
template <typename T, Flags f>
constexpr Angle<T, f> Angle<T, f>::from_clamped_rad(const T& v) {
  assert(v > -pi<float> && v <= pi<float>);
  return Angle<T, f>(v);
}
template <typename T, Flags f>
constexpr Angle<T, f> Angle<T, f>::from_clamped_deg(const T& v) {
  return from_clamped_rad(v / T(180) * pi<T>);
}
template <typename T, Flags f>
constexpr Angle<T, f> Angle<T, f>::from_rad(const T& v) {
  T constrained = cste::mod(v + pi<T>, two_pi<T>);
  if (constrained <= T(0)) {
    constrained += two_pi<T>;
  }
  constrained -= pi<T>;
  return from_clamped_rad(constrained);
}
template <typename T, Flags f>
constexpr Angle<T, f> Angle<T, f>::from_deg(const T& v) {
  return from_rad(v / T(180) * pi<T>);
}
template <typename T, Flags f>
constexpr T Angle<T, f>::as_deg() const {
  return value_ * T(180) / pi<T>;
}
template <typename T, Flags f>
constexpr T Angle<T, f>::as_rad() const {
  return value_;
}
template <typename T, Flags f>
constexpr const T& Angle<T, f>::raw() const {
  return value_;
}
}  // namespace VECPP_NAMESPACE

namespace VECPP_NAMESPACE {
template <typename T, Flags f>
constexpr T sin(const Angle<T, f>& a) {
  if constexpr (is_ct(f)) {
    double r = a.as_rad();
    bool neg = false;
    if(r < 0.0) {
      r *= -1.0;
      neg = true;
    }
    if(r > half_pi<double>) {
      r = pi<double> - r;
    }
    double r_2 = r*r * -1.0;
    double result = r;
    for (unsigned long long i = 3; i < 19; i+=2) {
      r *= r_2;
      result += r / cste::factorial(i);
    }
    if(neg) {
      result *= -1.0;
    }
    return T(result);
  } else {
    return std::sin(a.as_rad());
  }
}
template <typename T, Flags f>
constexpr T cos(const Angle<T, f>& a) {
  if constexpr (is_ct(f)) {
    return sin(a + Angle<T, f>::from_rad(half_pi<T>));
  } else {
    return std::cos(a.as_rad());
  }
}
template <typename T, Flags f>
constexpr T tan(const Angle<T, f>& a) {
  if constexpr (is_ct(f)) {
    return sin(a) / cos(a);
  } else {
    return std::tan(a.as_rad());
  }
}
}  // namespace VECPP_NAMESPACE

namespace VECPP_NAMESPACE {
template <typename T, std::size_t len, Flags f = 0>
struct Vec {
 public:
  using value_type = T;
  static constexpr Flags flags = f;
  static constexpr std::size_t size() { return len; }
  constexpr T& at(std::size_t i) {
    if (i >= len) {
      throw std::out_of_range("out of range vector access");
    }
    return data_[i];
  }
  constexpr const T& at(std::size_t i) const {
    if (i >= len) {
      throw std::out_of_range("out of range vector access");
    }
    return data_[i];
  }
  constexpr T& operator[](std::size_t i) {
    assert(i < len);
    return data_[i];
  }
  constexpr const T& operator[](std::size_t i) const {
    assert(i < len);
    return data_[i];
  }
  constexpr T* data() { return data_.data(); }
  constexpr const T* data() const { return data_.data(); }
  // Left public for aggregate initialization.
  std::array<T, len> data_;
  // A vector is implicitely convertible to any vector differing only by flags
  template <int new_flags>
  constexpr operator Vec<T, len, new_flags>() const {
    Vec<T, len, new_flags> result = {};
    for (std::size_t i = 0; i < size(); ++i) {
      result[i] = data_[i];
    }
    return result;
  }
};
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f | flags::compile_time> ct(const Vec<T, l, f>& v) {
  return v;
}
template <typename T, std::size_t l, Flags f>
constexpr T* begin(Vec<T, l, f>& v) {
  return v.data();
}
template <typename T, std::size_t l, Flags f>
constexpr T* end(Vec<T, l, f>& v) {
  return v.data() + v.size();
}
template <typename T, std::size_t l, Flags f>
constexpr const T* begin(const Vec<T, l, f>& v) {
  return v.data();
}
template <typename T, std::size_t l, Flags f>
constexpr const T* end(const Vec<T, l, f>& v) {
  return v.data() + v.size();
}
template <typename T, std::size_t l, Flags f>
std::ostream& operator<<(std::ostream& stream, const Vec<T, l, f>& vec) {
  stream << "(";
  bool first = true;
  for (const auto& v : vec) {
    if (!first) {
      stream << ", ";
    } else {
      first = false;
    }
    stream << v;
  }
  stream << ")";
  return stream;
}
template <typename T, std::size_t l, Flags f1, Flags f2>
constexpr bool operator==(const Vec<T, l, f1>& lhs, const Vec<T, l, f2>& rhs) {
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}
template <typename T, std::size_t l, Flags f1, Flags f2>
constexpr bool operator!=(const Vec<T, l, f1>& lhs, const Vec<T, l, f2>& rhs) {
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return true;
    }
  }
  return false;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> operator-(const Vec<T, l, f>& rhs) {
  Vec<T, l, f> result = {};
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    result[i] = -rhs[i];
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f>& operator+=(Vec<T, l, f>& lhs, const Vec<T, l, f>& rhs) {
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    lhs[i] += rhs[i];
  }
  return lhs;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> operator+(const Vec<T, l, f>& lhs,
                                 const Vec<T, l, f>& rhs) {
  Vec<T, l, f> result = lhs;
  result += rhs;
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f>& operator-=(Vec<T, l, f>& lhs, const Vec<T, l, f>& rhs) {
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    lhs[i] -= rhs[i];
  }
  return lhs;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> operator-(const Vec<T, l, f>& lhs,
                                 const Vec<T, l, f>& rhs) {
  Vec<T, l, f> result = lhs;
  result -= rhs;
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f>& operator*=(Vec<T, l, f>& lhs, const Vec<T, l, f>& rhs) {
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    lhs[i] *= rhs[i];
  }
  return lhs;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> operator*(const Vec<T, l, f>& lhs,
                                 const Vec<T, l, f>& rhs) {
  Vec<T, l, f> result = lhs;
  result *= rhs;
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f>& operator/=(Vec<T, l, f>& lhs, const Vec<T, l, f>& rhs) {
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    lhs[i] /= rhs[i];
  }
  return lhs;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> operator/(const Vec<T, l, f>& lhs,
                                 const Vec<T, l, f>& rhs) {
  Vec<T, l, f> result = lhs;
  result /= rhs;
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f>& operator*=(Vec<T, l, f>& lhs, const T& rhs) {
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    lhs[i] *= rhs;
  }
  return lhs;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> operator*(const Vec<T, l, f>& lhs, const T& rhs) {
  Vec<T, l, f> result = lhs;
  result *= rhs;
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> operator*(const T& lhs, const Vec<T, l, f>& rhs) {
  return rhs * lhs;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f>& operator/=(Vec<T, l, f>& lhs, const T& rhs) {
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    lhs[i] /= rhs;
  }
  return lhs;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> operator/(const Vec<T, l, f>& lhs, const T& rhs) {
  Vec<T, l, f> result = lhs;
  result /= rhs;
  return result;
}
}  // namespace VECPP_NAMESPACE

namespace VECPP_NAMESPACE {
template <typename T, Flags f>
constexpr Vec<T, 3, f> cross(const Vec<T, 3, f>& lhs, const Vec<T, 3, f>& rhs) {
  return {lhs[1] * rhs[2] - lhs[2] * rhs[1], lhs[2] * rhs[0] - lhs[0] * rhs[2],
          lhs[0] * rhs[1] - lhs[1] * rhs[0]};
}
template <typename T, std::size_t l, Flags f>
constexpr T dot(const Vec<T, l, f>& lhs, const Vec<T, l, f>& rhs) {
  T result = 0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    result += lhs[i] * rhs[i];
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr T norm(const Vec<T, l, f>& v) {
  return sqrt<f>(dot(v, v));
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> normalize(const Vec<T, l, f>& v) {
  return v / norm(v);
}
}  // namespace VECPP_NAMESPACE

namespace VECPP_NAMESPACE {
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> abs(const Vec<T, l, f>& vec) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < vec.size(); ++i) {
    result[i] = abs<f>(vec[i]);
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> ceil(const Vec<T, l, f>& v) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < v.size(); ++i) {
    result[i] = ceil<f>(v[i]);
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> floor(const Vec<T, l, f>& v) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < v.size(); ++i) {
    result[i] = floor<f>(v[i]);
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> fract(const Vec<T, l, f>& v) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < v.size(); ++i) {
    result[i] = fract<f>(v[i]);
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> round(const Vec<T, l, f>& v) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < v.size(); ++i) {
    result[i] = round<f>(v[i]);
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> sign(const Vec<T, l, f>& v) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < v.size(); ++i) {
    result[i] = sign<f>(v[i]);
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> trunc(const Vec<T, l, f>& v) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < v.size(); ++i) {
    result[i] = trunc<f>(v[i]);
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> max(const Vec<T, l, f>& lhs, const Vec<T, l, f>& rhs) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    result[i] = max(lhs[i], rhs[i]);
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> min(const Vec<T, l, f>& lhs, const Vec<T, l, f>& rhs) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    result[i] = min(lhs[i], rhs[i]);
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> mod(const Vec<T, l, f>& lhs, const Vec<T, l, f>& rhs) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    result[i] = mod<f>(lhs[i], rhs[i]);
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> step(const Vec<T, l, f>& lhs, const Vec<T, l, f>& rhs) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    result[i] = step<f>(lhs[i], rhs[i]);
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> clamp(const Vec<T, l, f>& v, const Vec<T, l, f>& low,
                             const Vec<T, l, f>& high) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < v.size(); ++i) {
    result[i] = clamp<f>(v[i], low[i], high[i]);
  }
  return result;
}
}  // namespace VECPP_NAMESPACE

namespace VECPP_NAMESPACE {
template <typename T, std::size_t C, std::size_t R, Flags f = 0>
struct Mat {
  static constexpr Flags flags = f;
  static constexpr std::size_t rows = R;
  static constexpr std::size_t cols = C;
  using value_type = T;
  using col_type = Vec<value_type, rows, flags>;
  using row_type = Vec<value_type, cols, flags>;
  constexpr value_type& operator()(std::size_t i, std::size_t j) {
    assert(i < cols && j < rows);
    return data_[i*rows + j];
  }
  constexpr const value_type& operator()(std::size_t i, std::size_t j) const {
    assert(i < cols && j < rows);
    return data_[i*rows + j];
  }
  constexpr value_type& at(std::size_t c, std::size_t r) {
    if (c >= cols || r >= rows) {
      throw std::out_of_range("out of range matrix access");
    }
    return (*this)(c, r);
  }
  constexpr const value_type& at(std::size_t c, std::size_t r) const {
    if (c >= cols || r >= rows) {
      throw std::out_of_range("out of range matrix access");
    }
    return (*this)(c, r);
  }
  constexpr value_type* data() {
    return data_.data();
  }
  constexpr const value_type* data() const {
    return data_.data();
  }
  // Left public for aggregate initialization.
  std::array<value_type, cols * rows> data_;
};
template <typename T, std::size_t C, std::size_t R, Flags fl, Flags fr>
constexpr bool operator==(const Mat<T, C, R, fl>& lhs, const Mat<T, C, R, fr>& rhs) {
  for(std::size_t i = 0 ; i < C; ++i) {
    for(std::size_t j = 0 ; j < R; ++j) {
      if(lhs(i, j) != rhs(i, j)) {
        return false;
      }
    }
  }
  return true;
}
template<typename T, std::size_t C, std::size_t R, Flags fl>
std::ostream& operator<<(std::ostream& stream, const Mat<T, C, R, fl>& lhs) {
  stream << "[";
  for(std::size_t i = 0; i < R; ++i) {
    stream << " ";
    for(std::size_t j = 0; j < C; ++j) {
      stream << lhs(i,j) << ",";
    }
    stream << "\n";
  }
  stream << "]";
  return stream;
}
template <typename T, std::size_t C, std::size_t R, Flags mf, Flags vf>
constexpr Vec<T, R, vf> operator*(const Mat<T, C, R, mf>& mat,
                                  const Vec<T, C, vf>& vec) {
  Vec<T, R, vf> result = {};
  for (std::size_t i = 0; i < R; ++i) {
    T v = 0;
    for (std::size_t j = 0; j < C; ++j) {
      v += mat(j,i) * vec[j];
    }
    result[i] = v;
  }
  return result;
}
template <typename T, std::size_t C, std::size_t R, Flags mf, Flags vf>
constexpr Vec<T, C, vf> operator*(const Vec<T, R, vf>& vec,
                                  const Mat<T, C, R, mf>& mat) {
  Vec<T, C, vf> result = {};
  for (std::size_t j = 0; j < C; ++j) {
    T v = 0;
    for (std::size_t i = 0; i < R; ++i) {
      v += mat(j,i) * vec[i];
    }
    result[j] = v;
  }
  return result;
}
}

namespace VECPP_NAMESPACE {
  // ***************** DETERMINANT ***************** //
  template<typename MatT>
  struct Mat_determinant;
  template<typename MatT>
  constexpr typename MatT::value_type determinant(const MatT& mat) {
    return Mat_determinant<MatT>::calc_determinant(mat);
  }
  // SPECIALIZATIONS:
  template<typename T, Flags f>
  struct Mat_determinant<Mat<T, 2, 2, f>> {
    using MatT = Mat<T, 2, 2, f>;
    static constexpr T calc_determinant(const MatT& mat) {
      return mat(0, 0) * mat(1, 1) - mat(1, 0) * mat(0, 1);
    }
  };
  template<typename T, Flags f>
  struct Mat_determinant<Mat<T, 3, 3, f>> {
    using MatT = Mat<T, 3, 3, f>;
    static constexpr T calc_determinant(const MatT& mat) {
      return
        mat(0, 0) * (mat(1, 1) * mat(2, 2) - mat(2, 1) * mat(1, 2)) -
        mat(1, 0) * (mat(0, 1) * mat(2, 2) - mat(2, 1) * mat(0, 2)) +
        mat(2, 0) * (mat(0, 1) * mat(1, 2) - mat(1, 1) * mat(0, 2));
    }
  };
  template<typename T, std::size_t N, Flags f>
  struct Mat_determinant <Mat<T, N, N, f>> {
    using MatT = Mat<T, N, N, f>;
    static constexpr T calc_determinant(const MatT& A) {
      T result = T(0);
      T sign = T(1);
      for(std::size_t i = 0; i < N; ++i) {
        Mat<T, N-1, N-1, f> cf = {};
        for(std::size_t j = 0; j < N-1; ++j) {
          for(std::size_t k = 0; k < N-1; ++k) {
            cf(j, k) = A(
                j < i ? j : j + 1,
                k + 1);
          }
        }
        result += sign * determinant(cf) * A(i, 0);
        sign = sign * T(-1);
      }
      return result;
    }
  };
  // ***************** TRANSPOSE ***************** //
  template <typename T, std::size_t C, std::size_t R, Flags f>
  constexpr Mat<T, R, C, f> transpose(const Mat<T, C, R, f>& m) {
    Mat<T, R, C, f> result = {};
    for(std::size_t i = 0 ; i < R; ++i) {
      for(std::size_t j = 0 ; j < C; ++j) {
        result(i, j) = m(j, i);
      }
    }
    return result;
  }
}

namespace VECPP_NAMESPACE {
template <typename T>
struct Quat {
  using value_type = T;
  template <Flags af>
  static constexpr Quat angle_axis(const Angle<T, af>& angle,
                                   const Vec<T, 3>& axis);
  // Left public for aggregate initialization.
  T w;
  T x;
  T y;
  T z;
};
template <typename T>
template <Flags af>
constexpr Quat<T> Quat<T>::angle_axis(const Angle<T, af>& angle,
                                      const Vec<T, 3>& axis) {
  const T s = sin(angle * T(0.5));
  const T c = cos(angle * T(0.5));
  return {c, axis[0] * s, axis[1] * s, axis[2] * s};
}
template <typename T>
constexpr Quat<T>& operator*=(Quat<T>& lhs, const Quat<T>& rhs) {
  const Quat<T> p(lhs);
  const Quat<T> q(rhs);
  lhs.w = p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z;
  lhs.x = p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y;
  lhs.y = p.w * q.y + p.y * q.w + p.z * q.x - p.x * q.z;
  lhs.z = p.w * q.z + p.z * q.w + p.x * q.y - p.y * q.x;
  return lhs;
}
template <typename T>
constexpr Quat<T> operator*(const Quat<T>& lhs, const Quat<T>& rhs) {
  Quat<T> result(lhs);
  result *= rhs;
  return result;
}
template <typename T>
constexpr Vec<T, 3> operator*(const Quat<T>& lhs, const Vec<T, 3>& rhs) {
  const Vec<T, 3> q_v = {lhs.x, lhs.y, lhs.z};
  const Vec<T, 3> uv = cross(q_v, rhs);
  const Vec<T, 3> uuv = cross(q_v, uv);
  return rhs + ((uv * lhs.w) + uuv) * T(2);
}
}  // namespace VECPP_NAMESPACE


#endif
