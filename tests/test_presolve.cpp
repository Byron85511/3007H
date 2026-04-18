#include <gtest/gtest.h>

#include "lp_solver/model/problem_data.hpp"
#include "lp_solver/presolve/presolver.hpp"
#include "lp_solver/util/packed_matrix.hpp"

using namespace lp_solver;

TEST(Presolve, DetectsEmptyRowInconsistency) {
  model::ProblemData prob{
      util::PackedMatrix::Builder(2, 2).appendColumn({1}, {1.0}).appendColumn({1}, {1.0}).build(),
      std::vector<double>{0.0, 0.0},
      std::vector<double>{3.0, 0.0},
      {},
      {},
  };
  const auto res = presolve::presolveStandardForm(prob);
  EXPECT_EQ(res.status, presolve::PresolveStatus::PrimalInfeasible);
}

TEST(Presolve, DetectsUnboundedEmptyColumn) {
  model::ProblemData prob{
      util::PackedMatrix::Builder(1, 2).appendColumn({0}, {1.0}).appendColumn({}, {}).build(),
      std::vector<double>{0.0, -1.0},
      std::vector<double>{1.0},
      {},
      {},
  };
  const auto res = presolve::presolveStandardForm(prob);
  EXPECT_EQ(res.status, presolve::PresolveStatus::Unbounded);
}
