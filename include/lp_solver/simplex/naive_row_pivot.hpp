#pragma once

#include "lp_solver/model/solver_state.hpp"
#include "lp_solver/simplex/i_row_pivot.hpp"

namespace lp_solver::simplex {

/// Classic dual row selection: first basic row with \(x_B < 0\).
class NaiveRowPivot final : public IRowPivot {
 public:
  explicit NaiveRowPivot(double primal_tol = 1e-7) : tol_(primal_tol) {}

  [[nodiscard]] int chooseRow(const model::SolverState& state) const override {
    const int m = static_cast<int>(state.x_basic.size());
    for (int i = 0; i < m; ++i) {
      if (state.x_basic[static_cast<size_t>(i)] < -tol_) {
        return i;
      }
    }
    return -1;
  }

 private:
  double tol_;
};

}  // namespace lp_solver::simplex
