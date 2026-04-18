#pragma once

#include <memory>

#include "lp_solver/linalg/i_basis_factor.hpp"
#include "lp_solver/model/problem_data.hpp"
#include "lp_solver/model/solver_state.hpp"
#include "lp_solver/util/packed_matrix.hpp"
#include "lp_solver/simplex/i_row_pivot.hpp"
#include "lp_solver/simplex/i_solver_observer.hpp"
#include "lp_solver/util/indexed_vector.hpp"

namespace lp_solver::simplex {

struct SolverConfig {
  int max_iterations{10000};
  int refactor_frequency{100};
  double dual_feasibility_tol{1e-7};
  double primal_feasibility_tol{1e-7};
  bool use_harris_ratio_test{true};
  bool use_dual_steepest_edge{false};
};

class DualSimplex {
 public:
  enum class Status { Optimal, Infeasible, Unbounded, DualInfeasible, IterationLimit };

  DualSimplex(std::unique_ptr<linalg::IBasisFactor> factor,
              std::unique_ptr<IRowPivot> row_pivot,
              ISolverObserver* observer = nullptr);

  [[nodiscard]] Status solve(const model::ProblemData& prob, model::SolverState& state,
                             const SolverConfig& cfg = {});

 private:
  [[nodiscard]] int chuzr(const model::SolverState& state) const;
  void btran(util::IndexedVector& ep) const;
  [[nodiscard]] int chuzc(const util::IndexedVector& pivot_row, const model::ProblemData& prob,
                          const model::SolverState& state, const SolverConfig& cfg) const;
  void ftran(util::IndexedVector& aq) const;
  void pivot(int leaving_row, int entering_col, const util::IndexedVector& ftran_col,
             const util::IndexedVector& btran_ep, const model::ProblemData& prob,
             model::SolverState& state, const SolverConfig& cfg);
  void refactorBasis(const model::ProblemData& prob, model::SolverState& state);
  void initializeDseWeights(const model::ProblemData& prob, model::SolverState& state) const;

  [[nodiscard]] static util::PackedMatrix buildBasisMatrix(const model::ProblemData& prob,
                                                           const model::SolverState& state);

  std::unique_ptr<linalg::IBasisFactor> factor_;
  std::unique_ptr<IRowPivot> row_pivot_;
  ISolverObserver* observer_;
};

}  // namespace lp_solver::simplex
