#include <gtest/gtest.h>

#include "lp_solver/linalg/i_basis_factor.hpp"
#include "lp_solver/model/initialization.hpp"
#include "lp_solver/model/problem_data.hpp"
#include "lp_solver/model/solver_state.hpp"
#include "lp_solver/simplex/dual_simplex.hpp"
#include "lp_solver/simplex/naive_row_pivot.hpp"

using namespace lp_solver;

// Big-M style model: after forcing pivot, dual-feasible slack basis with a negative slack.
// This integration test only checks the solver terminates optimally on a manually augmented system.
TEST(BigM, DualFeasibleNegativeSlackStillOptimizes) {
  model::ProblemData prob{
      util::PackedMatrix::Builder(2, 4)
          .appendColumn({0, 1}, {-1.0, 1.0})
          .appendColumn({1}, {1.0})
          .appendColumn({0}, {1.0})
          .appendColumn({1}, {1.0})
          .build(),
      std::vector<double>{0.0, 0.0, 0.0, 0.0},
      std::vector<double>{-1.0, 2.0},
      {},
      {},
  };

  model::SolverState state;
  model::initializeSlackBasis(prob, state);

  simplex::DualSimplex solver(linalg::makeFactor(linalg::FactorBackend::Eigen),
                              std::make_unique<simplex::NaiveRowPivot>());
  simplex::SolverConfig cfg;
  cfg.max_iterations = 100;
  cfg.refactor_frequency = 20;

  EXPECT_EQ(solver.solve(prob, state, cfg), simplex::DualSimplex::Status::Optimal);
}
