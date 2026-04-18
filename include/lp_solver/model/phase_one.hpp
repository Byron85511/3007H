#pragma once

#include "lp_solver/model/solver_state.hpp"

namespace lp_solver::model {

/// True if any reduced cost is below \p tol (dual infeasible in minimization form).
[[nodiscard]] inline bool hasNegativeReducedCost(const SolverState& state, double tol = 1e-7) {
  for (double rc : state.reduced_costs) {
    if (rc < -tol) {
      return true;
    }
  }
  return false;
}

}  // namespace lp_solver::model
