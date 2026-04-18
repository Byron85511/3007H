#pragma once

#include "lp_solver/model/problem_data.hpp"
#include "lp_solver/model/solver_state.hpp"

namespace lp_solver::model {

/// Assumes the last \p numRows() columns of \p A form an identity (slack) basis.
void initializeSlackBasis(const ProblemData& prob, SolverState& state);

void computeReducedCosts(const ProblemData& prob, const std::vector<double>& pi,
                         SolverState& state);

[[nodiscard]] double computeObjective(const ProblemData& prob, const SolverState& state);

}  // namespace lp_solver::model
