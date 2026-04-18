#include <gtest/gtest.h>

#include "lp_solver/linalg/i_basis_factor.hpp"
#include "lp_solver/model/initialization.hpp"
#include "lp_solver/model/problem_data.hpp"
#include "lp_solver/model/solver_state.hpp"
#include "lp_solver/simplex/dse_row_pivot.hpp"
#include "lp_solver/simplex/dual_simplex.hpp"
#include "lp_solver/simplex/naive_row_pivot.hpp"

using namespace lp_solver;

TEST(DualSimplex, SlackOptimalImmediately) {
  model::ProblemData prob{
      util::PackedMatrix::Builder(1, 3)
          .appendColumn({0}, {1.0})
          .appendColumn({0}, {1.0})
          .appendColumn({0}, {1.0})
          .build(),
      std::vector<double>{1.0, 1.0, 0.0},
      std::vector<double>{1.0},
      {},
      {},
  };

  model::SolverState state;
  model::initializeSlackBasis(prob, state);

  simplex::DualSimplex solver(linalg::makeFactor(linalg::FactorBackend::Eigen),
                              std::make_unique<simplex::NaiveRowPivot>());
  simplex::SolverConfig cfg;
  cfg.max_iterations = 50;
  cfg.refactor_frequency = 5;

  const auto status = solver.solve(prob, state, cfg);
  EXPECT_EQ(status, simplex::DualSimplex::Status::Optimal);
  EXPECT_NEAR(state.objective, 0.0, 1e-6);
}

TEST(DualSimplex, OneNegativeSlackPivots) {
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
  cfg.max_iterations = 50;
  cfg.refactor_frequency = 10;

  const auto status = solver.solve(prob, state, cfg);
  EXPECT_EQ(status, simplex::DualSimplex::Status::Optimal);
}

TEST(DualSimplex, DualSteepestEdgeMatchesNaiveOnTiny) {
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

  model::SolverState state_naive;
  model::initializeSlackBasis(prob, state_naive);
  simplex::SolverConfig cfg;
  cfg.max_iterations = 50;
  cfg.refactor_frequency = 10;
  cfg.use_dual_steepest_edge = false;

  simplex::DualSimplex naive_solver(linalg::makeFactor(linalg::FactorBackend::Eigen),
                                    std::make_unique<simplex::NaiveRowPivot>());
  ASSERT_EQ(naive_solver.solve(prob, state_naive, cfg), simplex::DualSimplex::Status::Optimal);

  model::SolverState state_dse;
  model::initializeSlackBasis(prob, state_dse);
  cfg.use_dual_steepest_edge = true;
  simplex::DualSimplex dse_solver(linalg::makeFactor(linalg::FactorBackend::Eigen),
                                  std::make_unique<simplex::DseRowPivot>());
  ASSERT_EQ(dse_solver.solve(prob, state_dse, cfg), simplex::DualSimplex::Status::Optimal);

  EXPECT_NEAR(state_naive.objective, state_dse.objective, 1e-8);
}

TEST(DualSimplex, ReportsDualInfeasibleWhenNoLeavingRowButNegativeReducedCosts) {
  model::ProblemData prob{
      util::PackedMatrix::Builder(1, 2).appendColumn({0}, {1.0}).appendColumn({0}, {1.0}).build(),
      std::vector<double>{-1.0, 0.0},
      std::vector<double>{1.0},
      {},
      {},
  };

  model::SolverState state;
  model::initializeSlackBasis(prob, state);

  simplex::DualSimplex solver(linalg::makeFactor(linalg::FactorBackend::Eigen),
                              std::make_unique<simplex::NaiveRowPivot>());
  simplex::SolverConfig cfg;
  cfg.max_iterations = 10;

  EXPECT_EQ(solver.solve(prob, state, cfg), simplex::DualSimplex::Status::DualInfeasible);
}
