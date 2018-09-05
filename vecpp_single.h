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
struct Scalar_traits {
  enum { force_constexpr = false };
};
template <typename T>
struct Vec_traits : public Scalar_traits {
  enum {
    align = alignof(T),
  };
};
template <typename T>
struct Mat_traits : public Scalar_traits {
  enum {
    align = alignof(T),
  };
};
template <typename Traits>
constexpr bool is_ct() {
  return Traits::force_constexpr;
}
template<typename Traits>
struct With_constexpr : public Traits {
  enum { force_constexpr = true };
};
template<typename Traits>
using Add_constexpr_t = std::conditional_t<
    is_ct<Traits>(),
    Traits,
    With_constexpr<Traits>
  >;
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
  if
    constexpr(std::is_integral_v<T>) { return v % d; }
  else {
    return std::fmod(v, d);
  }
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
constexpr T mod(const T& v, const T& d) {
  if
    constexpr(std::is_integral_v<T>) { return v % d; }
  else {
    return v - floor(v / d) * d;
  }
}
}  // namespace cste
template <typename traits = Scalar_traits, typename ScalarT>
constexpr ScalarT abs(const ScalarT& v) {
  return v < ScalarT(0) ? -v : v;
}
template <typename traits = Scalar_traits, typename ScalarT>
constexpr ScalarT ceil(const ScalarT& v) {
  if
    constexpr(is_ct<traits>()) { return cste::ceil(v); }
  else {
    return non_cste::ceil(v);
  }
}
template <typename traits = Scalar_traits, typename ScalarT>
constexpr ScalarT exp(const ScalarT& v) {
  if
    constexpr(is_ct<traits>()) { return cste::exp(v); }
  else {
    return non_cste::exp(v);
  }
}
constexpr unsigned long long factorial(std::size_t N) {
  unsigned long long result = 1;
  for (unsigned long long i = 1; i <= N; ++i) {
    result *= i;
  }
  return result;
}
template <typename traits = Scalar_traits, typename ScalarT>
constexpr ScalarT floor(const ScalarT& v) {
  if
    constexpr(is_ct<traits>()) { return cste::floor(v); }
  else {
    return non_cste::floor(v);
  }
}
template <typename traits = Scalar_traits, typename ScalarT>
constexpr ScalarT round(const ScalarT& v) {
  if
    constexpr(is_ct<traits>()) { return cste::round(v); }
  else {
    return non_cste::round(v);
  }
}
template <typename traits = Scalar_traits, typename ScalarT>
constexpr ScalarT sign(const ScalarT& v) {
  return v >= 0.0f ? 1.0f : -1.0f;
}
template <typename traits = Scalar_traits, typename ScalarT>
constexpr ScalarT trunc(const ScalarT& v) {
  if
    constexpr(is_ct<traits>()) { return cste::trunc(v); }
  else {
    return non_cste::trunc(v);
  }
}
template <typename traits = Scalar_traits, typename ScalarT>
constexpr ScalarT mod(const ScalarT& v, const ScalarT& d) {
  if
    constexpr(is_ct<traits>()) { return cste::mod(v, d); }
  else {
    return non_cste::mod(v, d);
  }
}
template <typename traits = Scalar_traits, typename ScalarT>
constexpr ScalarT fract(const ScalarT& v) {
  if
    constexpr(is_ct<traits>()) { return cste::fract(v); }
  else {
    return non_cste::fract(v);
  }
}
template <typename traits = Scalar_traits, typename ScalarT>
constexpr ScalarT step(const ScalarT& edge, const ScalarT& x) {
  return x < edge ? 0.0f : 1.0f;
}
template <typename traits = Scalar_traits, typename ScalarT>
constexpr ScalarT min(const ScalarT& lhs, const ScalarT& rhs) {
  return std::min(lhs, rhs);
}
template <typename traits = Scalar_traits, typename ScalarT>
constexpr ScalarT max(const ScalarT& lhs, const ScalarT& rhs) {
  return std::max(lhs, rhs);
}
template <typename traits = Scalar_traits, typename ScalarT>
constexpr ScalarT clamp(const ScalarT& v, const ScalarT& low,
                        const ScalarT& high) {
  return std::clamp(v, low, high);
}
template <typename traits = Scalar_traits, typename ScalarT, typename PctT>
constexpr ScalarT lerp(const ScalarT& from, const ScalarT& to,
                       const PctT& pct) {
  return from + (to - from) * pct;
}
template <typename traits = Scalar_traits, typename ScalarT>
constexpr ScalarT pow(const ScalarT& x, const ScalarT& n) {
  if
    constexpr(is_ct<traits>()) { return cste::pow(x, n); }
  else {
    return non_cste::pow(x, n);
  }
}
template <typename traits = Scalar_traits, typename T>
constexpr T sqrt(const T& v) {
  if
    constexpr(is_ct<traits>()) { return cste::sqrt(v); }
  else {
    return non_cste::sqrt(v);
  }
}
}  // namespace VECPP_NAMESPACE

namespace VECPP_NAMESPACE {
template <typename T, typename TraitsT = Scalar_traits>
class Angle {
 public:
  using value_type = T;
  using traits = TraitsT;
  static constexpr Angle from_rad(const value_type&);
  static constexpr Angle from_deg(const value_type&);
  // The argument MUST be in the ]-PI, PI] range.
  static constexpr Angle from_clamped_rad(const value_type&);
  // The argument MUST be in the ]-180, 180] range.
  static constexpr Angle from_clamped_deg(const value_type&);
  constexpr value_type as_deg() const;
  constexpr value_type as_rad() const;
  constexpr const value_type& raw() const;
  template <typename New_traits>
  constexpr operator Angle<T, New_traits>() const;
 private:
  value_type value_;
  // Constructs an angle from a constrained radian value.
  explicit constexpr Angle(const T&);
};
template <typename T, typename traits>
constexpr auto ct(const Angle<T, traits>& v) {
  return Angle<T, Add_constexpr_t<traits>>(v);
}
template <typename T, typename Traits>
template <typename New_traits>
constexpr Angle<T, Traits>::operator Angle<T, New_traits>() const {
  return Angle<T, New_traits>::from_clamped_rad(value_);
}
template <typename T, typename Traits>
constexpr Angle<T, Traits> operator-(const Angle<T, Traits>& rhs) {
  T value = rhs.as_rad();
  // Special case, we keep positive pi.
  if (value != pi<T>) {
    value = -value;
  }
  return Angle<T, Traits>::from_clamped_rad(value);
}
template <typename T, typename Traits>
constexpr Angle<T, Traits>& operator+=(Angle<T, Traits>& lhs,
                                       const Angle<T, Traits>& rhs) {
  T val = lhs.as_rad() + rhs.as_rad();
  // Since both lhs and rhs are in the ]-PI,PI] range, the sum is in the
  // ]-2PI-1,2PI] range, so we can make assumptions in the constraining process.
  if (val > pi<T>) {
    val -= two_pi<T>;
  } else if (val <= -pi<T>) {
    val += two_pi<T>;
  }
  lhs = Angle<T, Traits>::from_clamped_rad(val);
  return lhs;
}
template <typename T, typename Traits>
constexpr Angle<T, Traits> operator+(const Angle<T, Traits>& lhs,
                                     const Angle<T, Traits>& rhs) {
  auto result = lhs;
  result += rhs;
  return result;
}
template <typename T, typename Traits>
constexpr Angle<T, Traits>& operator-=(Angle<T, Traits>& lhs,
                                       const Angle<T, Traits>& rhs) {
  T val = lhs.as_rad() - rhs.as_rad();
  // Since both lhs and rhs are in the ]-PI,PI] range, the difference is in the
  // ]-2PI,2PI[ range, so we can make assumptions in the constraining process.
  if (val > pi<T>) {
    val -= two_pi<T>;
  } else if (val <= -pi<T>) {
    val += two_pi<T>;
  }
  lhs = Angle<T, Traits>::from_clamped_rad(val);
  return lhs;
}
template <typename T, typename Traits>
constexpr Angle<T, Traits> operator-(const Angle<T, Traits>& lhs,
                                     const Angle<T, Traits>& rhs) {
  auto result = lhs;
  result -= rhs;
  return result;
}
template <typename T, typename Traits>
constexpr Angle<T, Traits>& operator*=(Angle<T, Traits>& lhs, const T& rhs) {
  lhs = Angle<T, Traits>::from_rad(lhs.as_rad() * rhs);
  return lhs;
}
template <typename T, typename Traits>
constexpr Angle<T, Traits> operator*(const Angle<T, Traits>& lhs,
                                     const T& rhs) {
  auto result = lhs;
  result *= rhs;
  return result;
}
template <typename T, typename Traits>
constexpr Angle<T, Traits> operator*(const T& lhs,
                                     const Angle<T, Traits>& rhs) {
  return rhs * lhs;
}
template <typename T, typename Traits>
constexpr Angle<T, Traits>& operator/=(Angle<T, Traits>& lhs, const T& rhs) {
  lhs = Angle<T, Traits>::from_rad(lhs.as_rad() / rhs);
  return lhs;
}
template <typename T, typename Traits>
constexpr Angle<T, Traits> operator/(const Angle<T, Traits>& lhs,
                                     const T& rhs) {
  auto result = lhs;
  result /= rhs;
  return result;
}
template <typename T, typename Traits1, typename Traits2>
constexpr bool operator==(const Angle<T, Traits1>& lhs,
                          const Angle<T, Traits2>& rhs) {
  return lhs.raw() == rhs.raw();
}
template <typename T, typename Traits1, typename Traits2>
constexpr bool operator!=(const Angle<T, Traits1>& lhs,
                          const Angle<T, Traits2>& rhs) {
  return lhs.raw() != rhs.raw();
}
template <typename T, typename Traits1, typename Traits2>
constexpr bool operator<(const Angle<T, Traits1>& lhs,
                         const Angle<T, Traits2>& rhs) {
  return lhs.raw() < rhs.raw();
}
template <typename T, typename Traits1, typename Traits2>
constexpr bool operator>(const Angle<T, Traits1>& lhs,
                         const Angle<T, Traits2>& rhs) {
  return lhs.raw() > rhs.raw();
}
template <typename T, typename Traits1, typename Traits2>
constexpr bool operator<=(const Angle<T, Traits1>& lhs,
                          const Angle<T, Traits2>& rhs) {
  return lhs.raw() <= rhs.raw();
}
template <typename T, typename Traits1, typename Traits2>
constexpr bool operator>=(const Angle<T, Traits1>& lhs,
                          const Angle<T, Traits2>& rhs) {
  return lhs.raw() >= rhs.raw();
}
template <typename T, typename Traits>
std::ostream& operator<<(std::ostream& stream, const Angle<T, Traits>& v) {
  return stream << v.as_deg() << "°";
}
template <typename T, typename Traits>
constexpr Angle<T, Traits>::Angle(const T& v) : value_(v) {}
template <typename T, typename Traits>
constexpr Angle<T, Traits> Angle<T, Traits>::from_clamped_rad(const T& v) {
  assert(v > -pi<float> && v <= pi<float>);
  return Angle<T, Traits>(v);
}
template <typename T, typename Traits>
constexpr Angle<T, Traits> Angle<T, Traits>::from_clamped_deg(const T& v) {
  return from_clamped_rad(v / T(180) * pi<T>);
}
template <typename T, typename Traits>
constexpr Angle<T, Traits> Angle<T, Traits>::from_rad(const T& v) {
  T constrained = cste::mod(v + pi<T>, two_pi<T>);
  if (constrained <= T(0)) {
    constrained += two_pi<T>;
  }
  constrained -= pi<T>;
  return from_clamped_rad(constrained);
}
template <typename T, typename Traits>
constexpr Angle<T, Traits> Angle<T, Traits>::from_deg(const T& v) {
  return from_rad(v / T(180) * pi<T>);
}
template <typename T, typename Traits>
constexpr T Angle<T, Traits>::as_deg() const {
  return value_ * T(180) / pi<T>;
}
template <typename T, typename Traits>
constexpr T Angle<T, Traits>::as_rad() const {
  return value_;
}
template <typename T, typename Traits>
constexpr const T& Angle<T, Traits>::raw() const {
  return value_;
}
}  // namespace VECPP_NAMESPACE

namespace VECPP_NAMESPACE {
template <typename T, typename Traits>
constexpr T sin(const Angle<T, Traits>& a) {
  if
    constexpr(is_ct<Traits>()) {
      double r = a.as_rad();
      bool neg = false;
      if (r < 0.0) {
        r *= -1.0;
        neg = true;
      }
      if (r > half_pi<double>) {
        r = pi<double> - r;
      }
      double r_2 = r * r * -1.0;
      double result = r;
      for (std::size_t i = 3; i < 19; i += 2) {
        r *= r_2;
        result += r / factorial(i);
      }
      if (neg) {
        result *= -1.0;
      }
      return T(result);
    }
  else {
    return std::sin(a.as_rad());
  }
}
template <typename T, typename Traits>
constexpr T cos(const Angle<T, Traits>& a) {
  if
    constexpr(is_ct<Traits>()) {
      return sin(a + Angle<T, Traits>::from_rad(half_pi<T>));
    }
  else {
    return std::cos(a.as_rad());
  }
}
template <typename T, typename Traits>
constexpr T tan(const Angle<T, Traits>& a) {
  if
    constexpr(is_ct<Traits>()) { return sin(a) / cos(a); }
  else {
    return std::tan(a.as_rad());
  }
}
}  // namespace VECPP_NAMESPACE

namespace VECPP_NAMESPACE {
template <typename T, std::size_t len, typename Traits = Vec_traits<T>>
struct alignas(Traits::align) Vec {
 public:
  static_assert(len > 0);
  static constexpr std::size_t length = len;
  using value_type = T;
  using traits = Traits;
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
  template <typename new_traits>
  constexpr operator Vec<T, len, new_traits>() const {
    Vec<T, len, new_traits> result = {};
    for (std::size_t i = 0; i < length; ++i) {
      result[i] = data_[i];
    }
    return result;
  }
};
template <typename T, std::size_t l, typename traits>
constexpr auto ct(const Vec<T, l, traits>& v) {
  return Vec<T, l, Add_constexpr_t<traits>>(v);
}
template <typename T, std::size_t l, typename traits>
constexpr T* begin(Vec<T, l, traits>& v) {
  return v.data();
}
template <typename T, std::size_t l, typename traits>
constexpr T* end(Vec<T, l, traits>& v) {
  return v.data() + v.length;
}
template <typename T, std::size_t l, typename traits>
constexpr const T* begin(const Vec<T, l, traits>& v) {
  return v.data();
}
template <typename T, std::size_t l, typename traits>
constexpr const T* end(const Vec<T, l, traits>& v) {
  return v.data() + v.length;
}
template <typename T, std::size_t l, typename traits>
std::ostream& operator<<(std::ostream& stream, const Vec<T, l, traits>& vec) {
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
template <typename T, std::size_t l, typename traits1, typename traits2>
constexpr bool operator==(const Vec<T, l, traits1>& lhs,
                          const Vec<T, l, traits2>& rhs) {
  for (std::size_t i = 0; i < lhs.length; ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}
template <typename T, std::size_t l, typename traits1, typename traits2>
constexpr bool operator!=(const Vec<T, l, traits1>& lhs,
                          const Vec<T, l, traits2>& rhs) {
  for (std::size_t i = 0; i < lhs.length; ++i) {
    if (lhs[i] != rhs[i]) {
      return true;
    }
  }
  return false;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> operator-(const Vec<T, l, traits>& rhs) {
  Vec<T, l, traits> result = {};
  for (std::size_t i = 0; i < rhs.length; ++i) {
    result[i] = -rhs[i];
  }
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits>& operator+=(Vec<T, l, traits>& lhs,
                                        const Vec<T, l, traits>& rhs) {
  for (std::size_t i = 0; i < lhs.length; ++i) {
    lhs[i] += rhs[i];
  }
  return lhs;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> operator+(const Vec<T, l, traits>& lhs,
                                      const Vec<T, l, traits>& rhs) {
  Vec<T, l, traits> result = lhs;
  result += rhs;
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits>& operator-=(Vec<T, l, traits>& lhs,
                                        const Vec<T, l, traits>& rhs) {
  for (std::size_t i = 0; i < lhs.length; ++i) {
    lhs[i] -= rhs[i];
  }
  return lhs;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> operator-(const Vec<T, l, traits>& lhs,
                                      const Vec<T, l, traits>& rhs) {
  Vec<T, l, traits> result = lhs;
  result -= rhs;
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits>& operator*=(Vec<T, l, traits>& lhs,
                                        const Vec<T, l, traits>& rhs) {
  for (std::size_t i = 0; i < lhs.length; ++i) {
    lhs[i] *= rhs[i];
  }
  return lhs;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> operator*(const Vec<T, l, traits>& lhs,
                                      const Vec<T, l, traits>& rhs) {
  Vec<T, l, traits> result = lhs;
  result *= rhs;
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits>& operator/=(Vec<T, l, traits>& lhs,
                                        const Vec<T, l, traits>& rhs) {
  for (std::size_t i = 0; i < lhs.length; ++i) {
    lhs[i] /= rhs[i];
  }
  return lhs;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> operator/(const Vec<T, l, traits>& lhs,
                                      const Vec<T, l, traits>& rhs) {
  Vec<T, l, traits> result = lhs;
  result /= rhs;
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits>& operator*=(Vec<T, l, traits>& lhs, const T& rhs) {
  for (std::size_t i = 0; i < lhs.length; ++i) {
    lhs[i] *= rhs;
  }
  return lhs;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> operator*(const Vec<T, l, traits>& lhs,
                                      const T& rhs) {
  Vec<T, l, traits> result = lhs;
  result *= rhs;
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> operator*(const T& lhs,
                                      const Vec<T, l, traits>& rhs) {
  return rhs * lhs;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits>& operator/=(Vec<T, l, traits>& lhs, const T& rhs) {
  for (std::size_t i = 0; i < lhs.length; ++i) {
    lhs[i] /= rhs;
  }
  return lhs;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> operator/(const Vec<T, l, traits>& lhs,
                                      const T& rhs) {
  Vec<T, l, traits> result = lhs;
  result /= rhs;
  return result;
}
}  // namespace VECPP_NAMESPACE

namespace VECPP_NAMESPACE {
template <typename T, typename traits>
constexpr Vec<T, 3, traits> cross(const Vec<T, 3, traits>& lhs,
                                  const Vec<T, 3, traits>& rhs) {
  return {lhs[1] * rhs[2] - lhs[2] * rhs[1], lhs[2] * rhs[0] - lhs[0] * rhs[2],
          lhs[0] * rhs[1] - lhs[1] * rhs[0]};
}
template <typename T, std::size_t l, typename traits>
constexpr T dot(const Vec<T, l, traits>& lhs, const Vec<T, l, traits>& rhs) {
  T result = 0;
  for (std::size_t i = 0; i < lhs.length; ++i) {
    result += lhs[i] * rhs[i];
  }
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr T norm(const Vec<T, l, traits>& v) {
  return sqrt<traits>(dot(v, v));
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> normalize(const Vec<T, l, traits>& v) {
  return v / norm(v);
}
}  // namespace VECPP_NAMESPACE

namespace VECPP_NAMESPACE {
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> abs(const Vec<T, l, traits>& vec) {
  Vec<T, l, traits> result = {0};
  for (std::size_t i = 0; i < vec.length; ++i) {
    result[i] = abs<traits>(vec[i]);
  }
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> ceil(const Vec<T, l, traits>& v) {
  Vec<T, l, traits> result = {0};
  for (std::size_t i = 0; i < v.length; ++i) {
    result[i] = ceil<traits>(v[i]);
  }
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> floor(const Vec<T, l, traits>& v) {
  Vec<T, l, traits> result = {0};
  for (std::size_t i = 0; i < v.length; ++i) {
    result[i] = floor<traits>(v[i]);
  }
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> fract(const Vec<T, l, traits>& v) {
  Vec<T, l, traits> result = {0};
  for (std::size_t i = 0; i < v.length; ++i) {
    result[i] = fract<traits>(v[i]);
  }
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> round(const Vec<T, l, traits>& v) {
  Vec<T, l, traits> result = {0};
  for (std::size_t i = 0; i < v.length; ++i) {
    result[i] = round<traits>(v[i]);
  }
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> sign(const Vec<T, l, traits>& v) {
  Vec<T, l, traits> result = {0};
  for (std::size_t i = 0; i < v.length; ++i) {
    result[i] = sign<traits>(v[i]);
  }
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> trunc(const Vec<T, l, traits>& v) {
  Vec<T, l, traits> result = {0};
  for (std::size_t i = 0; i < v.length; ++i) {
    result[i] = trunc<traits>(v[i]);
  }
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> max(const Vec<T, l, traits>& lhs,
                                const Vec<T, l, traits>& rhs) {
  Vec<T, l, traits> result = {0};
  for (std::size_t i = 0; i < lhs.length; ++i) {
    result[i] = max(lhs[i], rhs[i]);
  }
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> min(const Vec<T, l, traits>& lhs,
                                const Vec<T, l, traits>& rhs) {
  Vec<T, l, traits> result = {0};
  for (std::size_t i = 0; i < lhs.length; ++i) {
    result[i] = min(lhs[i], rhs[i]);
  }
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> mod(const Vec<T, l, traits>& lhs,
                                const Vec<T, l, traits>& rhs) {
  Vec<T, l, traits> result = {0};
  for (std::size_t i = 0; i < lhs.length; ++i) {
    result[i] = mod<traits>(lhs[i], rhs[i]);
  }
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> step(const Vec<T, l, traits>& lhs,
                                 const Vec<T, l, traits>& rhs) {
  Vec<T, l, traits> result = {0};
  for (std::size_t i = 0; i < lhs.length; ++i) {
    result[i] = step<traits>(lhs[i], rhs[i]);
  }
  return result;
}
template <typename T, std::size_t l, typename traits>
constexpr Vec<T, l, traits> clamp(const Vec<T, l, traits>& v,
                                  const Vec<T, l, traits>& low,
                                  const Vec<T, l, traits>& high) {
  Vec<T, l, traits> result = {0};
  for (std::size_t i = 0; i < v.length; ++i) {
    result[i] = clamp<traits>(v[i], low[i], high[i]);
  }
  return result;
}
}  // namespace VECPP_NAMESPACE

namespace VECPP_NAMESPACE {
template <typename T, std::size_t C, std::size_t R,
          typename Traits = Mat_traits<T>>
struct Mat {
  static_assert(C > 0 && R > 0);
  static constexpr std::size_t rows = R;
  static constexpr std::size_t cols = C;
  using value_type = T;
  using traits = Traits;
  constexpr value_type& operator()(std::size_t i, std::size_t j) {
    assert(i < cols && j < rows);
    return data_[i * rows + j];
  }
  constexpr const value_type& operator()(std::size_t i, std::size_t j) const {
    assert(i < cols && j < rows);
    return data_[i * rows + j];
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
  constexpr value_type* data() { return data_.data(); }
  constexpr const value_type* data() const { return data_.data(); }
  // Left public for aggregate initialization.
  std::array<value_type, cols * rows> data_;
};
template <typename T, std::size_t C, std::size_t R, typename L_traits,
          typename R_traits>
constexpr bool operator==(const Mat<T, C, R, L_traits>& lhs,
                          const Mat<T, C, R, R_traits>& rhs) {
  for (std::size_t i = 0; i < C; ++i) {
    for (std::size_t j = 0; j < R; ++j) {
      if (lhs(i, j) != rhs(i, j)) {
        return false;
      }
    }
  }
  return true;
}
template <typename T, std::size_t C, std::size_t R, typename Traits>
std::ostream& operator<<(std::ostream& stream,
                         const Mat<T, C, R, Traits>& lhs) {
  stream << "[";
  for (std::size_t i = 0; i < R; ++i) {
    stream << " ";
    for (std::size_t j = 0; j < C; ++j) {
      stream << lhs(i, j) << ",";
    }
    stream << "\n";
  }
  stream << "]";
  return stream;
}
template <typename T, std::size_t C, std::size_t R, typename M_traits,
          typename V_traits>
constexpr Vec<T, R, V_traits> operator*(const Mat<T, C, R, M_traits>& mat,
                                        const Vec<T, C, V_traits>& vec) {
  Vec<T, R, V_traits> result = {};
  for (std::size_t i = 0; i < R; ++i) {
    T v = 0;
    for (std::size_t j = 0; j < C; ++j) {
      v += mat(j, i) * vec[j];
    }
    result[i] = v;
  }
  return result;
}
template <typename T, std::size_t C, std::size_t R, typename M_traits,
          typename V_traits>
constexpr Vec<T, C, V_traits> operator*(const Vec<T, R, V_traits>& vec,
                                        const Mat<T, C, R, M_traits>& mat) {
  Vec<T, C, V_traits> result = {};
  for (std::size_t j = 0; j < C; ++j) {
    T v = 0;
    for (std::size_t i = 0; i < R; ++i) {
      v += mat(j, i) * vec[i];
    }
    result[j] = v;
  }
  return result;
}
}

namespace VECPP_NAMESPACE {
template <typename MatT>
struct Mat_determinant;
template <typename MatT>
constexpr typename MatT::value_type determinant(const MatT& mat) {
  return Mat_determinant<MatT>::calc_determinant(mat);
}
template <typename T, typename Traits>
struct Mat_determinant<Mat<T, 2, 2, Traits>> {
  using MatT = Mat<T, 2, 2, Traits>;
  static constexpr T calc_determinant(const MatT& mat) {
    return mat(0, 0) * mat(1, 1) - mat(1, 0) * mat(0, 1);
  }
};
template <typename T, typename Traits>
struct Mat_determinant<Mat<T, 3, 3, Traits>> {
  using MatT = Mat<T, 3, 3, Traits>;
  static constexpr T calc_determinant(const MatT& mat) {
    return mat(0, 0) * (mat(1, 1) * mat(2, 2) - mat(2, 1) * mat(1, 2)) -
           mat(1, 0) * (mat(0, 1) * mat(2, 2) - mat(2, 1) * mat(0, 2)) +
           mat(2, 0) * (mat(0, 1) * mat(1, 2) - mat(1, 1) * mat(0, 2));
  }
};
template <typename T, std::size_t N, typename Traits>
struct Mat_determinant<Mat<T, N, N, Traits>> {
  using MatT = Mat<T, N, N, Traits>;
  static constexpr T calc_determinant(const MatT& A) {
    T result = T(0);
    T sign = T(1);
    for (std::size_t i = 0; i < N; ++i) {
      Mat<T, N - 1, N - 1, Traits> cf = {};
      for (std::size_t j = 0; j < N - 1; ++j) {
        for (std::size_t k = 0; k < N - 1; ++k) {
          cf(j, k) = A(j < i ? j : j + 1, k + 1);
        }
      }
      result += sign * determinant(cf) * A(i, 0);
      sign = sign * T(-1);
    }
    return result;
  }
};
template <typename T, std::size_t C, std::size_t R, typename Traits>
constexpr Mat<T, R, C, Traits> transpose(const Mat<T, C, R, Traits>& m) {
  Mat<T, R, C, Traits> result = {};
  for (std::size_t i = 0; i < R; ++i) {
    for (std::size_t j = 0; j < C; ++j) {
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
  template <typename A_traits>
  static constexpr Quat angle_axis(const Angle<T, A_traits>& angle,
                                   const Vec<T, 3>& axis);
  // Left public for aggregate initialization.
  T w;
  T x;
  T y;
  T z;
};
template <typename T>
template <typename A_traits>
constexpr Quat<T> Quat<T>::angle_axis(const Angle<T, A_traits>& angle,
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
template <typename T, typename V_traits>
constexpr Vec<T, 3, V_traits> operator*(const Quat<T>& lhs,
                                        const Vec<T, 3, V_traits>& rhs) {
  const Vec<T, 3, V_traits> q_v = {lhs.x, lhs.y, lhs.z};
  const Vec<T, 3, V_traits> uv = cross(q_v, rhs);
  const Vec<T, 3, V_traits> uuv = cross(q_v, uv);
  return rhs + ((uv * lhs.w) + uuv) * T(2);
}
}  // namespace VECPP_NAMESPACE


#endif
