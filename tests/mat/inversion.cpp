#include "catch.hpp"

#ifdef VECPP_TEST_SINGLE_HEADER
#include "vecpp/vecpp_single.h"
#else
#include "vecpp/vecpp.h"
#endif

using Catch::Matchers::WithinAbs;

TEST_CASE("Simple 2x2 mat inversion", "[mat][invert]") {
  using Mat2 = vecpp::Mat<float, 2, 2>;

  Mat2 m = {
    4, 7,
    2, 6
  };

  REQUIRE(vecpp::is_invertible(m));
  auto inverted = inverse(m);

  REQUIRE_THAT(inverted(0,0), WithinAbs(0.6f, 0.0001f));
  REQUIRE_THAT(inverted(0,1), WithinAbs(-0.7f, 0.0001f));
  REQUIRE_THAT(inverted(1,0), WithinAbs(-0.2f, 0.0001f));
  REQUIRE_THAT(inverted(1,1), WithinAbs(0.4f, 0.0001f));
}


TEST_CASE("Simple 3x3 mat inversion", "[mat][invert]") {
  using Mat3 = vecpp::Mat<float, 3, 3>;

  Mat3 m = {
    3, 0, 2,
    2, 0, -2,
    0, 1, 1
  };

  REQUIRE(vecpp::is_invertible(m));
  auto inverted = inverse(m);

  Mat3 expected_m = {
    0.2f, 0.2f, 0.0f,
    -0.2f, 0.3f, 1.0f,
    0.2f, -0.3f, 0.0f
  };

  for(int i = 0 ; i < 3; ++i) {
    for(int j = 0 ; j < 3; ++j) {
      REQUIRE_THAT(inverted(i, j), WithinAbs(expected_m(i,j), 0.0001f));    
    }
  }
}