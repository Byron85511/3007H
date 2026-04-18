#include "lp_solver/linalg/gilbert_peierls.hpp"

#include <unordered_set>

namespace lp_solver::linalg {

namespace {

void dfsPost(int u, const std::vector<std::vector<int>>& children, std::vector<char>& seen,
             std::vector<int>& post) {
  if (u < 0 || static_cast<size_t>(u) >= seen.size()) {
    return;
  }
  if (seen[static_cast<size_t>(u)]) {
    return;
  }
  seen[static_cast<size_t>(u)] = 1;
  for (int v : children[static_cast<size_t>(u)]) {
    dfsPost(v, children, seen, post);
  }
  post.push_back(u);
}

}  // namespace

std::vector<int> dfsPostorderFromSeeds(const std::vector<std::vector<int>>& children,
                                       const std::vector<int>& seeds) {
  const int n = static_cast<int>(children.size());
  std::vector<char> seen(static_cast<size_t>(n), 0);
  std::vector<int> post;
  std::unordered_set<int> uniq(seeds.begin(), seeds.end());
  for (int s : uniq) {
    if (s >= 0 && s < n) {
      dfsPost(s, children, seen, post);
    }
  }
  return post;
}

std::vector<int> forwardSubstitutionOrder(const std::vector<std::vector<int>>& children,
                                          const std::vector<int>& seeds) {
  const auto post = dfsPostorderFromSeeds(children, seeds);
  return std::vector<int>(post.rbegin(), post.rend());
}

}  // namespace lp_solver::linalg
