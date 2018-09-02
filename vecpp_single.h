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
T sin(const T& v) {
  return std::sin(v);
}
template <typename T>
T cos(const T& v) {
  return std::cos(v);
}
template <typename T>
T tan(const T& v) {
  return std::tan(v);
}
template <typename T>
constexpr T fmod(const T& v, const T& d) {
  return std::fmod(v, d);
}
}  // namespace non_cste
namespace cste {
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
constexpr T sin(const T& v) {
  assert(false);
}
template <typename T>
constexpr T cos(const T& v) {
  assert(false);
}
template <typename T>
constexpr T tan(const T& v) {
  return sin(v) / cos(v);
}
template <typename T>
constexpr T fmod(const T& v, const T& d) {
  return v - floor(v / d) * d;
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
constexpr ScalarT fmod(const ScalarT& v, const ScalarT& d) {
  return v - floor<f>(v / d) * d;
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
  constexpr value_type raw() const;
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
  T constrained = cste::fmod(v + pi<T>, two_pi<T>);
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
constexpr T Angle<T, f>::raw() const {
  return value_;
}
}  // namespace VECPP_NAMESPACE

namespace VECPP_NAMESPACE {
template <typename T, Flags f>
constexpr T sin(const Angle<T, f>& a) {
  if constexpr (is_ct(f)) {
    constexpr std::array<T, 5> taylor_factors = {-6, 120, -5040, 362880,
                                                 -39916800};
    T r = a.as_rad();
    T r_2 = r * r;
    T result = r;
    for (auto factor : taylor_factors) {
      r *= r_2;
      result += r / factor;
    }
    return result;
  } else {
    return non_cste::sin(a.as_rad());
  }
}
template <typename T, Flags f>
constexpr T cos(const Angle<T, f>& a) {
  if constexpr (is_ct(f)) {
    return sin(a + Angle<T, f>::from_rad(half_pi<T>));
  } else {
    return non_cste::cos(a.as_rad());
  }
}
template <typename T, Flags f>
constexpr T tan(const Angle<T, f>& a) {
  if constexpr (is_ct(f)) {
    return sin(a) / cos(a);
  } else {
    return non_cste::tan(a.as_rad());
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
constexpr T length(const Vec<T, l, f>& v) {
  return sqrt<f>(dot(v, v));
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> normalize(const Vec<T, l, f>& v) {
  return v / length(v);
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
    result[i] = std::max(lhs[i], rhs[i]);
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> min(const Vec<T, l, f>& lhs, const Vec<T, l, f>& rhs) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    result[i] = std::min(lhs[i], rhs[i]);
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> mod(const Vec<T, l, f>& lhs, const Vec<T, l, f>& rhs) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    result[i] = mod(lhs[i], rhs[i]);
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> step(const Vec<T, l, f>& lhs, const Vec<T, l, f>& rhs) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    result[i] = step(lhs[i], rhs[i]);
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> clamp(const Vec<T, l, f>& v, const Vec<T, l, f>& min,
                             const Vec<T, l, f>& max) {
  Vec<T, l, f> result = {0};
  for (std::size_t i = 0; i < v.size(); ++i) {
    result[i] = clamp(v[i], min[i], max[i]);
  }
  return result;
}
template <typename T, std::size_t l, Flags f>
constexpr Vec<T, l, f> lerp(const Vec<T, l, f>& from, const Vec<T, l, f>& to,
                            const T& pct) {
  return from + (to - from) * pct;
}
}  // namespace VECPP_NAMESPACE

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
