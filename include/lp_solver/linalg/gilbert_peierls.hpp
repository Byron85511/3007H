#pragma once

#include <vector>

namespace lp_solver::linalg {

/// Phase 1 of Gilbert–Peierls: DFS postorder on the DAG of a lower-triangular
/// nonzero pattern (column \p j influences rows \p children[j]).
[[nodiscard]] std::vector<int> dfsPostorderFromSeeds(const std::vector<std::vector<int>>& children,
                                                     const std::vector<int>& seeds);

/// Reverse postorder gives a valid forward-substitution order (manual §Hypersparsity).
[[nodiscard]] std::vector<int> forwardSubstitutionOrder(const std::vector<std::vector<int>>& children,
                                                      const std::vector<int>& seeds);

}  // namespace lp_solver::linalg
