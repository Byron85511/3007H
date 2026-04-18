#pragma once

#include <vector>

#include "lp_solver/model/problem_data.hpp"

namespace lp_solver::presolve {

enum class PresolveStatus { Ok, PrimalInfeasible, Unbounded };

struct PresolveResult {
  model::ProblemData reduced;
  PresolveStatus status{PresolveStatus::Ok};
};

/// Lightweight algebraic presolve for standard-form LPs (manual Phase 0).
[[nodiscard]] PresolveResult presolveStandardForm(const model::ProblemData& prob);

/// Map a solution of the reduced problem back to the original indexing (primal only).
void postsolvePrimal(const PresolveResult& meta, std::vector<double>& x_full);

}  // namespace lp_solver::presolve
