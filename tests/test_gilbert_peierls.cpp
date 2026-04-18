#include <gtest/gtest.h>

#include "lp_solver/linalg/gilbert_peierls.hpp"

using namespace lp_solver;

// Example DAG from MAT3007H Project Manual (5x5 L, rhs nonzero only at index 0).
TEST(GilbertPeierls, ManualExampleTopologicalOrder) {
  std::vector<std::vector<int>> children(5);
  children[0] = {2};
  children[2] = {4};

  const std::vector<int> seeds{0};
  const auto order = linalg::forwardSubstitutionOrder(children, seeds);
  ASSERT_EQ(order.size(), 3u);
  EXPECT_EQ(order[0], 0);
  EXPECT_EQ(order[1], 2);
  EXPECT_EQ(order[2], 4);
}
