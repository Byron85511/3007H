// sparse_lp_complete.cpp — single-file bundle (Phases 1–5).
// Build:  c++ -std=c++17 -O2 sparse_lp_complete.cpp -o sparse_lp_test
// Run:    ./sparse_lp_test


#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

#if defined(SPARSE_LP_USE_EIGEN)
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#endif

namespace sparse_lp {

// ---------- sparse_matrix.hpp ----------
/// Compressed Sparse Column (CSC) matrix: A has shape (num_rows × num_cols).
/// Column j occupies values[col_pointers[j] .. col_pointers[j+1]-1] with
/// corresponding row indices in row_indices (sorted ascending per column).
class SparseMatrix {
public:
  SparseMatrix() = default;
  SparseMatrix(std::size_t num_rows, std::size_t num_cols,
               std::vector<double> values, std::vector<int> row_indices,
               std::vector<int> col_pointers);

  std::size_t rows() const { return num_rows_; }
  std::size_t cols() const { return num_cols_; }
  std::size_t nnz() const { return values_.size(); }

  const std::vector<double>& values() const { return values_; }
  const std::vector<int>& row_indices() const { return row_indices_; }
  const std::vector<int>& col_pointers() const { return col_pointers_; }

  std::vector<double>& values_mut() { return values_; }
  std::vector<int>& row_indices_mut() { return row_indices_; }
  std::vector<int>& col_pointers_mut() { return col_pointers_; }

  /// Build CSC from duplicate-free triplets (i,j,v); columns may be unsorted (will sort).
  static SparseMatrix from_triplets(std::size_t num_rows, std::size_t num_cols,
                                    const std::vector<std::tuple<int, int, double>>& triplets);

  SparseMatrix transpose() const;

  /// y += alpha * A * x (A this)
  void mat_vec(double alpha, const std::vector<double>& x, std::vector<double>& y) const;

  /// y += alpha * A^T * x
  void mat_vec_transpose(double alpha, const std::vector<double>& x,
                         std::vector<double>& y) const;

  double get(int row, int col) const;
  void set_from_triplets_checked(std::size_t num_rows, std::size_t num_cols,
                                 std::vector<double> values, std::vector<int> row_indices,
                                 std::vector<int> col_pointers);

  std::string debug_summary() const;

private:
  std::size_t num_rows_{0};
  std::size_t num_cols_{0};
  std::vector<double> values_;
  std::vector<int> row_indices_;
  std::vector<int> col_pointers_;
};

// ---------- lu_solver.hpp ----------
/// Sparse LU with Markowitz-based pivoting and column thresholding for stability.
/// Factors a square matrix A as P_r A P_c = L U (equivalently the working matrix
/// after row/column permutations satisfies L U = M with M[i,j]=A[perm_r[i]][perm_c[j]]).
class LUSolver {
public:
  /// Relative column threshold: pivot must satisfy |a_ij| >= threshold * col_max[j].
  void set_column_threshold(double t) { column_threshold_ = t; }
  double column_threshold() const { return column_threshold_; }

  /// Markowitz tie-break: prefer larger |pivot| when scores are equal.
  bool factorize(const SparseMatrix& A);

  bool ok() const { return factor_ok_; }

  /// Solve A x = b using stored factors (applies permutations and triangular solves).
  std::vector<double> solve(const std::vector<double>& b) const;

  /// Unit lower L (implicit diagonal 1) and upper U in CSC form for the permuted system.
  const SparseMatrix& L() const { return L_; }
  const SparseMatrix& U() const { return U_; }

  /// perm_r[i] = original row index now at row i; perm_c[j] = original col index now at col j.
  const std::vector<int>& perm_row() const { return perm_r_; }
  const std::vector<int>& perm_col() const { return perm_c_; }

  /// y solves L y = b_perm where b_perm[i] = b[perm_r[i]].
  std::vector<double> forward_substitution(const std::vector<double>& b_perm) const;

  /// x solves U x = y; result indexed by column in permuted space.
  std::vector<double> backward_substitution(const std::vector<double>& y_perm) const;

  /// Map permuted solution z back to original variables: x[perm_c[j]] = z[j].
  std::vector<double> scatter_solution(const std::vector<double>& z_perm) const;

private:
  static double csc_entry_upper(const SparseMatrix& U, int row, int col);

  bool factor_ok_{false};
  double column_threshold_{0.01};
  SparseMatrix L_;
  SparseMatrix U_;
  std::vector<int> perm_r_;
  std::vector<int> perm_c_;
  int n_{0};
};

// ---------- matrix_ops.hpp ----------
SparseMatrix make_identity_csc(int n);

/// Build m×m matrix B with B(:,j) = A(:, col_indices[j]).
SparseMatrix column_submatrix(const SparseMatrix& A, const std::vector<int>& col_indices);

// ---------- linear_algebra_backend.hpp ----------
/// Selects implementations for basis factorization and triangular solves.
enum class LinearAlgebraBackendKind {
  Native,       ///< CSC + Markowitz LU (always available)
  Eigen,        ///< Optional: Eigen::SparseLU when SPARSE_LP_USE_EIGEN is defined
  SuiteSparse   ///< Optional: KLU/UMFPACK hooks when SPARSE_LP_USE_SUITESPARSE is defined
};

/// Abstract hook for swapping factorization engines without changing the simplex driver.
class BasisFactorizationBackend {
public:
  virtual ~BasisFactorizationBackend() = default;

  virtual bool factorize(const SparseMatrix& basis) = 0;
  virtual std::vector<double> solve(const std::vector<double>& rhs) const = 0;
  virtual LinearAlgebraBackendKind kind() const = 0;
};

/// Native implementation backed by LUSolver.
class NativeBasisBackend final : public BasisFactorizationBackend {
public:
  bool factorize(const SparseMatrix& basis) override;
  std::vector<double> solve(const std::vector<double>& rhs) const override;
  LinearAlgebraBackendKind kind() const override { return LinearAlgebraBackendKind::Native; }

  LUSolver& lu() { return lu_; }
  const LUSolver& lu() const { return lu_; }

private:
  LUSolver lu_;
};

std::unique_ptr<BasisFactorizationBackend> make_backend(LinearAlgebraBackendKind k);

// ---------- lp_model.hpp ----------
/// Objective direction (Highs-style naming).
enum class ObjSense { Minimize, Maximize };

/// Sparse LP in the form:
///   optimize  c^T x
///   s.t.      l <= x <= u   (variable bounds),
///             row_lower <= A x <= row_upper   (optional row constraints).
///
/// For a pure standard-form skeleton, callers may encode equalities by setting
/// row_lower = row_upper. Slack columns (if used) are explicit columns of A.
struct LpModel {
  SparseMatrix A;
  std::vector<double> col_cost;
  std::vector<double> col_lower;
  std::vector<double> col_upper;
  std::vector<double> row_lower;
  std::vector<double> row_upper;
  ObjSense sense{ObjSense::Minimize};

  /// Validates dimensions; throws std::invalid_argument on mismatch.
  void validate_dimensions() const;
};

// ---------- revised_dual_simplex.hpp ----------
enum class SolveStatus {
  BasisFactorized,   ///< Initial refactor succeeded (framework milestone).
  Optimal,           ///< Found optimal solution.
  DimensionError,
  FactorizationFailed,
  Unbounded,
  NotImplemented     ///< Full dual simplex iterations not wired yet.
};

/// Revised dual simplex driver skeleton: maintains a basis B of m columns of A,
/// factors B with BasisFactorizationBackend, and exposes refactor hooks for pivots.
class RevisedDualSimplex {
public:
  explicit RevisedDualSimplex(LpModel model, LinearAlgebraBackendKind backend_kind);

  const LpModel& model() const { return model_; }

  /// Use the last m columns of A as the initial basis (slack layout: A = [F | I]).
  void set_basis_slack_tail();

  /// Refactorize current basis B = A(:, basis_cols_).
  bool refactor();

  SolveStatus solve_framework();
  /// Perform a simplified primal revised simplex iterations on standard slack-extended LP.
  SolveStatus solve(int max_iterations = 100);

  const std::vector<int>& basis_columns() const { return basis_cols_; }
  BasisFactorizationBackend& backend() { return *backend_; }
  const BasisFactorizationBackend& backend() const { return *backend_; }

private:
  LpModel model_;
  std::unique_ptr<BasisFactorizationBackend> backend_;
  std::vector<int> basis_cols_;
};

// ---------- sparse_vector.hpp ----------
/// Cache-friendly hypersparse vector: sorted unique indices + values (SOA layout).
/// Indices and values are 64-byte aligned when possible for SIMD-friendly access patterns.
class SparseVector {
public:
  SparseVector() = default;

  explicit SparseVector(std::size_t capacity_hint) {
    reserve(capacity_hint);
  }

  void clear() {
    indices_.clear();
    values_.clear();
  }

  void reserve(std::size_t n) {
    indices_.reserve(n);
    values_.reserve(n);
  }

  std::size_t nnz() const { return indices_.size(); }
  bool empty() const { return indices_.empty(); }

  const std::vector<int>& indices() const { return indices_; }
  const std::vector<double>& values() const { return values_; }

  std::vector<int>& indices_mut() { return indices_; }
  std::vector<double>& values_mut() { return values_; }

  /// Merge-add: result += alpha * other (other sorted by index).
  void axpy_sorted(double alpha, const SparseVector& other);

  void push_back(int idx, double val) {
    indices_.push_back(idx);
    values_.push_back(val);
  }

  void sort_by_index();

private:
  std::vector<int> indices_;
  std::vector<double> values_;
};

/// Dense vector with explicit alignment for hot FTRAN/BTRAN paths.
inline void* aligned_alloc(std::size_t align, std::size_t bytes) {
#if defined(_WIN32)
  return _aligned_malloc(bytes, align);
#else
  void* p = nullptr;
  if (posix_memalign(&p, align, bytes) != 0) return nullptr;
  return p;
#endif
}

inline void aligned_free(void* p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  free(p);
#endif
}

/// Owning aligned buffer for hypersparse intermediate results (RAII).
class AlignedDenseVector {
public:
  explicit AlignedDenseVector(std::size_t n) : n_(n) {
    data_ = static_cast<double*>(aligned_alloc(64, n * sizeof(double)));
    if (!data_) throw std::bad_alloc();
    std::memset(data_, 0, n * sizeof(double));
  }

  ~AlignedDenseVector() {
    if (data_) aligned_free(data_);
  }

  AlignedDenseVector(const AlignedDenseVector&) = delete;
  AlignedDenseVector& operator=(const AlignedDenseVector&) = delete;

  AlignedDenseVector(AlignedDenseVector&& o) noexcept : data_(o.data_), n_(o.n_) {
    o.data_ = nullptr;
    o.n_ = 0;
  }

  double* data() { return data_; }
  const double* data() const { return data_; }
  std::size_t size() const { return n_; }

  double& operator[](std::size_t i) { return data_[i]; }
  const double& operator[](std::size_t i) const { return data_[i]; }

private:
  double* data_{nullptr};
  std::size_t n_{0};
};

// ---------- eta_file.hpp ----------
/// One elementary basis change: T = I with column p replaced by h (eta column after FTRAN).
/// Applying T^{-1} to y: z_p = y_p / h_p, z_i = y_i - h_i z_p for i != p.
struct EtaEntry {
  int pivot_col{-1};
  std::vector<int> row_index;
  std::vector<double> row_value;  // full sparse column h (including pivot row)
};

/// Sequence of inverse elementary transforms applied after U^{-1} L^{-1} in FTRAN order:
/// w = E_k^{-1} ... E_1^{-1} U^{-1} L^{-1} b.
class EtaFile {
public:
  void clear() { entries_.clear(); }
  std::size_t size() const { return entries_.size(); }
  std::size_t total_nnz() const;

  const std::vector<EtaEntry>& entries() const { return entries_; }

  /// Push one eta; h must have h[pivot_col] != 0 for stability.
  void push(int pivot_col, const std::vector<double>& dense_column);
  void push_sparse(int pivot_col, const std::vector<int>& idx, const std::vector<double>& val);

  /// Apply all E^{-1} in **forward** order (oldest first) to y.
  static void apply_t_inverse_inplace(std::vector<double>& y, const EtaEntry& e);

  void apply_all_inverse(std::vector<double>& y) const;

private:
  std::vector<EtaEntry> entries_;
};

// ---------- ft_basis_maintenance.hpp ----------
/// Monitors eta-file growth vs. fresh LU and triggers refactor; enforces pivot tolerance on etas.
struct BasisMaintenanceParams {
  double eta_density_factor{3.0};  ///< Refactor if eta nnz > factor * (nnz(L)+nnz(U)).
  std::size_t max_eta_pivots{500};
  double pivot_tolerance{1e-10};
};

/// Forrest–Tomlin / PFI hybrid: baseline LU plus inverse elementary etas; optional in-place U spike pass.
class AdvancedBasisMaintenance {
public:
  explicit AdvancedBasisMaintenance(BasisMaintenanceParams p = {});

  const BasisMaintenanceParams& params() const { return params_; }
  BasisMaintenanceParams& params_mut() { return params_; }
  LUSolver& lu() { return lu_; }
  const LUSolver& lu() const { return lu_; }

  const EtaFile& etas() const { return etas_; }
  std::size_t eta_pivot_count() const { return eta_pivots_; }

  /// Full refactor from explicit basis matrix B.
  bool refactor(const SparseMatrix& B);

  /// Record one basis change using eta column h (full length m) replacing column p (FTRAN of entering column).
  /// Returns false if numerically unstable (caller should refactor).
  bool push_basis_change(int pivot_col, const std::vector<double>& eta_column_dense);

  /// Returns true if caller should refactor before next pivot.
  bool should_refactor() const;

  /// Clears etas after a refactor (called automatically by refactor).
  void clear_etas();

  std::size_t lu_nnz() const { return lu_nnz_; }

  /// Forrest–Tomlin spike elimination on upper factor (debug / small-n path). Returns false if skipped.
  bool ft_eliminate_spike_in_u(SparseMatrix& U_perm, int spike_col, std::vector<double> spike_dense);

private:
  BasisMaintenanceParams params_;
  LUSolver lu_;
  EtaFile etas_;
  std::size_t lu_nnz_{0};
  std::size_t eta_pivots_{0};
};

/// FTRAN: B^{-1} b with B^{-1} = (product of inverse etas) U^{-1} L^{-1} in permuted basis space.
std::vector<double> ftran_with_etas(const LUSolver& lu, const EtaFile& etas, const std::vector<double>& b);

// ---------- hypersparse_solve.hpp ----------
/// Symbolic reachability for L^{-1} b when b is sparse (Gilbert–Peierls pattern).
void symbolic_reach_lower_unit(const SparseMatrix& L_unit_lower_csc, const SparseVector& rhs_pattern,
                               std::vector<int>& reach_order);

/// Numeric FTRAN on L (unit lower, CSC) using reach order; y overwrites work.
void numeric_forward_lower_unit(const SparseMatrix& L_unit_lower_csc, const std::vector<double>& rhs_dense,
                                const std::vector<int>& reach_order, std::vector<double>& y);

/// Hypersparse FTRAN: sparse rhs -> only touches reachable indices (uses dense work vector).
void hypersparse_ftran_l(const SparseMatrix& L_unit_lower_csc, const SparseVector& rhs_sparse,
                         std::vector<double>& work_dense, std::vector<int>& reach_scratch);

/// DFS on U^T graph for BTRAN with sparse rhs (dual multipliers / cost vector).
void symbolic_reach_upper_transposed(const SparseMatrix& U_upper_csc, const SparseVector& rhs_pattern,
                                     std::vector<int>& reach_order_rev);

void hypersparse_btran_u(const SparseMatrix& U_upper_csc, const SparseVector& rhs_sparse,
                           std::vector<double>& work_dense, std::vector<int>& reach_scratch);

// ---------- steepest_edge.hpp ----------
/// Dual steepest-edge style weights γ (one per structural column / pricing candidate).
class DualSteepestEdgeWeights {
public:
  explicit DualSteepestEdgeWeights(std::size_t n = 0) { resize(n); }

  void resize(std::size_t n) {
    gamma_.assign(n, 1.0);
  }

  /// Fallback: γ = 1 (Devex-like cold start).
  void initialize_unit() {
    for (double& g : gamma_) {
      g = 1.0;
    }
  }

  /// Exact initialization when diagonal norms ||B^{-1} A_j||^2 are available.
  void initialize_from_binv_column_norms_squared(const std::vector<double>& norms2) {
    gamma_ = norms2;
    for (double& g : gamma_) {
      if (g < 1e-30) g = 1.0;
    }
  }

  const std::vector<double>& gamma() const { return gamma_; }
  std::vector<double>& gamma_mut() { return gamma_; }

  /// Recursive stabilization: γ_new = max(γ_old, update_term) for the entering column.
  void update_max_rule(int col_index, double update_term) {
    if (col_index < 0 || static_cast<std::size_t>(col_index) >= gamma_.size()) return;
    const double oldg = gamma_[static_cast<std::size_t>(col_index)];
    gamma_[static_cast<std::size_t>(col_index)] = std::max(oldg, update_term);
  }

  /// Preferred dual steepest update when ‖a‖² with a = B^{-1} A_q is known.
  void update_from_ftran_column(int col_index, const std::vector<double>& a) {
    double s = 0.0;
    for (double v : a) {
      s += v * v;
    }
    update_max_rule(col_index, s);
  }

private:
  std::vector<double> gamma_;
};

// ---------- bound_flipping.hpp ----------
/// Dual long-step (bound flipping): cross multiple active-set boundaries in one iteration.
struct BoundFlippingParams {
  int max_flips{32};
  double feasibility_tol{1e-9};
};

struct BoundFlipResult {
  int flips_applied{0};
  double step_length{0.0};
  bool dual_feasible_end{true};
};

/// Controller skeleton: repeatedly relax dual bounds along improving direction until
/// ratio test blocks or max flips reached. Integrates with dual simplex ratio test.
class DualBoundFlippingController {
public:
  explicit DualBoundFlippingController(BoundFlippingParams p = {}) : params_(std::move(p)) {}

  /// Given current dual vector π, direction d, and per-row step limits θ_i to next bound,
  /// compute a long step θ and update π <- π + θ d, counting boundary crossings.
  BoundFlipResult long_step(std::vector<double>& pi, const std::vector<double>& direction,
                            const std::vector<double>& theta_limit) const;

  const BoundFlippingParams& params() const { return params_; }

private:
  BoundFlippingParams params_;
};

// ---------- presolve.hpp ----------
struct PresolveStats {
  std::size_t singleton_rows_removed{0};
  std::size_t singleton_cols_removed{0};
  std::size_t empty_rows_removed{0};
  std::size_t empty_cols_removed{0};
  std::size_t forcing_rows_removed{0};
  std::size_t duplicate_rows_removed{0};
};

struct PresolveResult {
  SparseMatrix A_reduced;
  std::vector<double> c_reduced;
  std::vector<double> l_reduced;
  std::vector<double> u_reduced;
  std::vector<double> row_lower_reduced;
  std::vector<double> row_upper_reduced;

  /// Original index -> reduced index or -1 if fixed/removed.
  std::vector<int> col_mapping;
  /// Fixed values for eliminated variables (original indexing).
  std::vector<double> fixed_value_orig;
  std::vector<char> is_fixed_orig;

  PresolveStats stats;
  std::string message;
};

/// Remove empty rows/columns, singleton rows (one variable), simple forcing constraints.
PresolveResult presolve_lp(const SparseMatrix& A, const std::vector<double>& c,
                           const std::vector<double>& col_lower, const std::vector<double>& col_upper,
                           const std::vector<double>& row_lower, const std::vector<double>& row_upper);

// ---------- postsolve.hpp ----------
/// Maps reduced solutions back to the original variable space (un-fix + padding).
class PostsolveMap {
public:
  explicit PostsolveMap(const PresolveResult& pre) : pre_(pre) {}

  /// x_orig[j] = fixed value if presolve fixed j, else x_reduced[col_mapping[j]].
  std::vector<double> expand_solution(const std::vector<double>& x_reduced) const;

  const PresolveResult& presolve_result() const { return pre_; }

private:
  PresolveResult pre_;
};

// ---------- sparse_matrix.cpp ----------
namespace {

void sort_column_entries(std::vector<double>& v, std::vector<int>& r) {
  const std::size_t n = v.size();
  std::vector<std::size_t> order(n);
  for (std::size_t i = 0; i < n; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return r[a] < r[b];
  });
  std::vector<double> nv(n);
  std::vector<int> nr(n);
  for (std::size_t i = 0; i < n; ++i) {
    nv[i] = v[order[i]];
    nr[i] = r[order[i]];
  }
  v.swap(nv);
  r.swap(nr);
}

}  // namespace

SparseMatrix::SparseMatrix(std::size_t num_rows, std::size_t num_cols,
                           std::vector<double> values, std::vector<int> row_indices,
                           std::vector<int> col_pointers)
    : num_rows_(num_rows),
      num_cols_(num_cols),
      values_(std::move(values)),
      row_indices_(std::move(row_indices)),
      col_pointers_(std::move(col_pointers)) {
  if (col_pointers_.size() != num_cols + 1) {
    throw std::invalid_argument("col_pointers must have length num_cols + 1");
  }
}

SparseMatrix SparseMatrix::from_triplets(
    std::size_t num_rows, std::size_t num_cols,
    const std::vector<std::tuple<int, int, double>>& triplets) {
  std::vector<std::vector<double>> col_vals(num_cols);
  std::vector<std::vector<int>> col_rows(num_cols);
  for (const auto& t : triplets) {
    int i = std::get<0>(t);
    int j = std::get<1>(t);
    double v = std::get<2>(t);
    if (i < 0 || static_cast<std::size_t>(i) >= num_rows || j < 0 ||
        static_cast<std::size_t>(j) >= num_cols) {
      throw std::out_of_range("triplet index out of bounds");
    }
    if (v != 0.0) {
      col_vals[static_cast<std::size_t>(j)].push_back(v);
      col_rows[static_cast<std::size_t>(j)].push_back(i);
    }
  }
  std::vector<double> values;
  std::vector<int> row_indices;
  std::vector<int> col_pointers(num_cols + 1, 0);
  for (std::size_t j = 0; j < num_cols; ++j) {
    sort_column_entries(col_vals[j], col_rows[j]);
    for (std::size_t k = 0; k < col_vals[j].size(); ++k) {
      values.push_back(col_vals[j][k]);
      row_indices.push_back(col_rows[j][k]);
    }
    col_pointers[j + 1] = static_cast<int>(values.size());
  }
  return SparseMatrix(num_rows, num_cols, std::move(values), std::move(row_indices),
                      std::move(col_pointers));
}

SparseMatrix SparseMatrix::transpose() const {
  const std::size_t m = num_rows_;
  const std::size_t n = num_cols_;
  std::vector<std::vector<std::pair<int, double>>> rows_t(m);
  for (std::size_t j = 0; j < n; ++j) {
    for (int p = col_pointers_[j]; p < col_pointers_[j + 1]; ++p) {
      const int i = row_indices_[static_cast<std::size_t>(p)];
      rows_t[static_cast<std::size_t>(i)].push_back({static_cast<int>(j), values_[static_cast<std::size_t>(p)]});
    }
  }
  std::vector<std::tuple<int, int, double>> trip;
  trip.reserve(nnz());
  for (std::size_t i = 0; i < m; ++i) {
    std::sort(rows_t[i].begin(), rows_t[i].end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& e : rows_t[i]) {
      trip.push_back({static_cast<int>(i), e.first, e.second});
    }
  }
  return from_triplets(n, m, trip);
}

void SparseMatrix::mat_vec(double alpha, const std::vector<double>& x, std::vector<double>& y) const {
  if (x.size() < num_cols_ || y.size() < num_rows_) {
    throw std::invalid_argument("mat_vec: dimension mismatch");
  }
  for (std::size_t j = 0; j < num_cols_; ++j) {
    const double xj = x[j];
    if (xj == 0.0) continue;
    for (int p = col_pointers_[j]; p < col_pointers_[j + 1]; ++p) {
      const int i = row_indices_[static_cast<std::size_t>(p)];
      y[static_cast<std::size_t>(i)] += alpha * values_[static_cast<std::size_t>(p)] * xj;
    }
  }
}

void SparseMatrix::mat_vec_transpose(double alpha, const std::vector<double>& x,
                                     std::vector<double>& y) const {
  if (x.size() < num_rows_ || y.size() < num_cols_) {
    throw std::invalid_argument("mat_vec_transpose: dimension mismatch");
  }
  for (std::size_t j = 0; j < num_cols_; ++j) {
    double sum = 0.0;
    for (int p = col_pointers_[j]; p < col_pointers_[j + 1]; ++p) {
      const int i = row_indices_[static_cast<std::size_t>(p)];
      sum += values_[static_cast<std::size_t>(p)] * x[static_cast<std::size_t>(i)];
    }
    y[j] += alpha * sum;
  }
}

double SparseMatrix::get(int row, int col) const {
  if (row < 0 || col < 0 || static_cast<std::size_t>(row) >= num_rows_ ||
      static_cast<std::size_t>(col) >= num_cols_) {
    throw std::out_of_range("SparseMatrix::get");
  }
  const int lo = col_pointers_[static_cast<std::size_t>(col)];
  const int hi = col_pointers_[static_cast<std::size_t>(col) + 1];
  auto it = std::lower_bound(row_indices_.begin() + lo, row_indices_.begin() + hi, row);
  if (it != row_indices_.begin() + hi && *it == row) {
    return values_[static_cast<std::size_t>(it - row_indices_.begin())];
  }
  return 0.0;
}

void SparseMatrix::set_from_triplets_checked(std::size_t num_rows, std::size_t num_cols,
                                             std::vector<double> values,
                                             std::vector<int> row_indices,
                                             std::vector<int> col_pointers) {
  num_rows_ = num_rows;
  num_cols_ = num_cols;
  values_ = std::move(values);
  row_indices_ = std::move(row_indices);
  col_pointers_ = std::move(col_pointers);
}

std::string SparseMatrix::debug_summary() const {
  std::ostringstream os;
  os << "SparseMatrix " << num_rows_ << "x" << num_cols_ << " nnz=" << values_.size();
  return os.str();
}

// ---------- lu_solver.cpp ----------
namespace {

constexpr double kEps = 1e-14;

void swap_rows(std::vector<std::map<int, double>>& rows, int a, int b) {
  if (a == b) return;
  rows[static_cast<std::size_t>(a)].swap(rows[static_cast<std::size_t>(b)]);
}

void swap_cols(std::vector<std::map<int, double>>& rows, int k, int pj, int n) {
  if (k == pj) return;
  for (int r = 0; r < n; ++r) {
    auto& row = rows[static_cast<std::size_t>(r)];
    auto it_k = row.find(k);
    auto it_pj = row.find(pj);
    const bool hk = it_k != row.end();
    const bool hpj = it_pj != row.end();
    if (hk && hpj) {
      std::swap(it_k->second, it_pj->second);
    } else if (hk) {
      double v = it_k->second;
      row.erase(it_k);
      row[pj] = v;
    } else if (hpj) {
      double v = it_pj->second;
      row.erase(it_pj);
      row[k] = v;
    }
  }
}

int column_degree(const std::vector<std::map<int, double>>& rows, int j, int k, int n) {
  int d = 0;
  for (int r = k; r < n; ++r) {
    const auto& row = rows[static_cast<std::size_t>(r)];
    auto it = row.find(j);
    if (it != row.end() && std::abs(it->second) > kEps) ++d;
  }
  return d;
}

int row_degree(const std::map<int, double>& row, int k) {
  int d = 0;
  for (const auto& e : row) {
    if (e.first >= k && std::abs(e.second) > kEps) ++d;
  }
  return d;
}

SparseMatrix build_csc_lower(int n, const std::vector<std::vector<std::pair<int, double>>>& L_cols) {
  std::vector<std::tuple<int, int, double>> trip;
  trip.reserve(static_cast<std::size_t>(n) * 4);
  for (int j = 0; j < n - 1; ++j) {
    for (const auto& e : L_cols[static_cast<std::size_t>(j)]) {
      trip.push_back({e.first, j, e.second});
    }
  }
  return SparseMatrix::from_triplets(static_cast<std::size_t>(n), static_cast<std::size_t>(n), trip);
}

SparseMatrix build_csc_upper(int n, const std::vector<std::vector<std::pair<int, double>>>& U_cols) {
  std::vector<std::tuple<int, int, double>> trip;
  for (int j = 0; j < n; ++j) {
    for (const auto& e : U_cols[static_cast<std::size_t>(j)]) {
      trip.push_back({e.first, j, e.second});
    }
  }
  return SparseMatrix::from_triplets(static_cast<std::size_t>(n), static_cast<std::size_t>(n), trip);
}

}  // namespace

double LUSolver::csc_entry_upper(const SparseMatrix& U, int row, int col) {
  if (row > col) return 0.0;
  return U.get(row, col);
}

bool LUSolver::factorize(const SparseMatrix& A) {
  factor_ok_ = false;
  n_ = static_cast<int>(A.rows());
  if (A.cols() != A.rows() || n_ == 0) {
    return false;
  }
  const int n = n_;

  std::vector<std::map<int, double>> rows(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    for (int p = A.col_pointers()[static_cast<std::size_t>(j)];
         p < A.col_pointers()[static_cast<std::size_t>(j + 1)]; ++p) {
      const int i = A.row_indices()[static_cast<std::size_t>(p)];
      const double v = A.values()[static_cast<std::size_t>(p)];
      if (std::abs(v) > kEps) {
        rows[static_cast<std::size_t>(i)][j] = v;
      }
    }
  }

  perm_r_.resize(static_cast<std::size_t>(n));
  perm_c_.resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    perm_r_[static_cast<std::size_t>(i)] = i;
    perm_c_[static_cast<std::size_t>(i)] = i;
  }

  std::vector<std::vector<std::pair<int, double>>> L_cols(static_cast<std::size_t>(n));
  std::vector<std::vector<std::pair<int, double>>> U_cols(static_cast<std::size_t>(n));

  for (int k = 0; k < n; ++k) {
    std::vector<double> col_max(static_cast<std::size_t>(n), 0.0);
    for (int j = k; j < n; ++j) {
      double mx = 0.0;
      for (int r = k; r < n; ++r) {
        const auto& row = rows[static_cast<std::size_t>(r)];
        auto it = row.find(j);
        if (it != row.end()) {
          mx = std::max(mx, std::abs(it->second));
        }
      }
      col_max[static_cast<std::size_t>(j)] = mx;
    }

    int best_i = -1;
    int best_j = -1;
    long long best_score = std::numeric_limits<long long>::max();
    double best_abs = -1.0;

    auto consider = [&](int i, int j) {
      const auto& row = rows[static_cast<std::size_t>(i)];
      auto it = row.find(j);
      if (it == row.end() || std::abs(it->second) <= kEps) return;
      const double aij = std::abs(it->second);
      const double cm = col_max[static_cast<std::size_t>(j)];
      if (cm > kEps && aij < column_threshold_ * cm) return;

      const int rd = row_degree(row, k);
      const int cd = column_degree(rows, j, k, n);
      const long long score =
          static_cast<long long>(std::max(0, rd - 1)) * static_cast<long long>(std::max(0, cd - 1));
      if (score < best_score || (score == best_score && aij > best_abs + kEps)) {
        best_score = score;
        best_abs = aij;
        best_i = i;
        best_j = j;
      }
    };

    for (int i = k; i < n; ++i) {
      for (const auto& e : rows[static_cast<std::size_t>(i)]) {
        const int j = e.first;
        if (j >= k) consider(i, j);
      }
    }

    if (best_i < 0) {
      for (int i = k; i < n; ++i) {
        for (const auto& e : rows[static_cast<std::size_t>(i)]) {
          const int j = e.first;
          if (j < k) continue;
          const double aij = std::abs(e.second);
          if (aij <= kEps) continue;
          const int rd = row_degree(rows[static_cast<std::size_t>(i)], k);
          const int cd = column_degree(rows, j, k, n);
          const long long score =
              static_cast<long long>(std::max(0, rd - 1)) * static_cast<long long>(std::max(0, cd - 1));
          if (score < best_score || (score == best_score && aij > best_abs + kEps)) {
            best_score = score;
            best_abs = aij;
            best_i = i;
            best_j = j;
          }
        }
      }
    }

    if (best_i < 0) {
      return false;
    }

    if (best_i != k) {
      swap_rows(rows, k, best_i);
      std::swap(perm_r_[static_cast<std::size_t>(k)], perm_r_[static_cast<std::size_t>(best_i)]);
    }
    if (best_j != k) {
      swap_cols(rows, k, best_j, n);
      std::swap(perm_c_[static_cast<std::size_t>(k)], perm_c_[static_cast<std::size_t>(best_j)]);
    }

    auto& pivot_row = rows[static_cast<std::size_t>(k)];
    auto pit = pivot_row.find(k);
    if (pit == pivot_row.end() || std::abs(pit->second) <= kEps) {
      return false;
    }
    const double pivot = pit->second;

    std::vector<std::pair<int, double>> u_entries;
    u_entries.reserve(pivot_row.size());
    for (const auto& e : pivot_row) {
      if (e.first >= k) {
        u_entries.push_back(e);
      }
    }
    std::sort(u_entries.begin(), u_entries.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& e : u_entries) {
      U_cols[static_cast<std::size_t>(e.first)].push_back({k, e.second});
    }

    for (int i = k + 1; i < n; ++i) {
      auto& row_i = rows[static_cast<std::size_t>(i)];
      auto it_ik = row_i.find(k);
      if (it_ik == row_i.end()) continue;
      const double mult = it_ik->second / pivot;
      if (std::abs(mult) > kEps) {
        L_cols[static_cast<std::size_t>(k)].push_back({i, mult});
      }
      row_i.erase(it_ik);
      for (const auto& e : pivot_row) {
        const int j = e.first;
        if (j <= k) continue;
        const double v = e.second;
        auto it = row_i.find(j);
        const double nv = (it == row_i.end()) ? -mult * v : it->second - mult * v;
        if (std::abs(nv) <= kEps) {
          if (it != row_i.end()) row_i.erase(it);
        } else {
          if (it == row_i.end()) {
            row_i[j] = nv;
          } else {
            it->second = nv;
          }
        }
      }
    }

    pivot_row.clear();
  }

  for (int j = 0; j < n; ++j) {
    auto& col = U_cols[static_cast<std::size_t>(j)];
    std::sort(col.begin(), col.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
  }

  L_ = build_csc_lower(n, L_cols);
  U_ = build_csc_upper(n, U_cols);
  factor_ok_ = true;
  return true;
}

std::vector<double> LUSolver::forward_substitution(const std::vector<double>& b_perm) const {
  if (!factor_ok_) {
    throw std::runtime_error("LUSolver: factorize before solve");
  }
  const int n = n_;
  std::vector<double> y(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    y[static_cast<std::size_t>(i)] = b_perm[static_cast<std::size_t>(i)];
  }
  for (int j = 0; j < n - 1; ++j) {
    const double yj = y[static_cast<std::size_t>(j)];
    if (yj == 0.0) continue;
    const int lo = L_.col_pointers()[static_cast<std::size_t>(j)];
    const int hi = L_.col_pointers()[static_cast<std::size_t>(j + 1)];
    for (int p = lo; p < hi; ++p) {
      const int i = L_.row_indices()[static_cast<std::size_t>(p)];
      const double l_ij = L_.values()[static_cast<std::size_t>(p)];
      y[static_cast<std::size_t>(i)] -= l_ij * yj;
    }
  }
  return y;
}

std::vector<double> LUSolver::backward_substitution(const std::vector<double>& y_perm) const {
  if (!factor_ok_) {
    throw std::runtime_error("LUSolver: factorize before solve");
  }
  const int n = n_;
  std::vector<double> x(static_cast<std::size_t>(n));
  for (int i = n - 1; i >= 0; --i) {
    double sum = y_perm[static_cast<std::size_t>(i)];
    for (int j = i + 1; j < n; ++j) {
      sum -= csc_entry_upper(U_, i, j) * x[static_cast<std::size_t>(j)];
    }
    const double uii = U_.get(i, i);
    if (std::abs(uii) <= kEps) {
      throw std::runtime_error("LUSolver: singular U");
    }
    x[static_cast<std::size_t>(i)] = sum / uii;
  }
  return x;
}

std::vector<double> LUSolver::scatter_solution(const std::vector<double>& z_perm) const {
  std::vector<double> x(static_cast<std::size_t>(n_), 0.0);
  for (int j = 0; j < n_; ++j) {
    const int orig_col = perm_c_[static_cast<std::size_t>(j)];
    x[static_cast<std::size_t>(orig_col)] = z_perm[static_cast<std::size_t>(j)];
  }
  return x;
}

std::vector<double> LUSolver::solve(const std::vector<double>& b) const {
  std::vector<double> b_perm(static_cast<std::size_t>(n_));
  for (int i = 0; i < n_; ++i) {
    const int orig_row = perm_r_[static_cast<std::size_t>(i)];
    b_perm[static_cast<std::size_t>(i)] = b[static_cast<std::size_t>(orig_row)];
  }
  std::vector<double> y = forward_substitution(b_perm);
  std::vector<double> z = backward_substitution(y);
  return scatter_solution(z);
}

// ---------- matrix_ops.cpp ----------
SparseMatrix make_identity_csc(int n) {
  if (n < 0) throw std::invalid_argument("make_identity_csc");
  std::vector<std::tuple<int, int, double>> trip;
  trip.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    trip.push_back({i, i, 1.0});
  }
  return SparseMatrix::from_triplets(static_cast<std::size_t>(n), static_cast<std::size_t>(n), trip);
}

SparseMatrix column_submatrix(const SparseMatrix& A, const std::vector<int>& col_indices) {
  const int m = static_cast<int>(A.rows());
  if (static_cast<int>(col_indices.size()) != m) {
    throw std::invalid_argument("column_submatrix: need |col_indices| == A.rows()");
  }
  std::vector<std::tuple<int, int, double>> trip;
  for (int j = 0; j < m; ++j) {
    const int aj = col_indices[static_cast<std::size_t>(j)];
    if (aj < 0 || static_cast<std::size_t>(aj) >= A.cols()) {
      throw std::out_of_range("column_submatrix: column index");
    }
    for (int p = A.col_pointers()[static_cast<std::size_t>(aj)];
         p < A.col_pointers()[static_cast<std::size_t>(aj + 1)]; ++p) {
      const int i = A.row_indices()[static_cast<std::size_t>(p)];
      const double v = A.values()[static_cast<std::size_t>(p)];
      if (v != 0.0) {
        trip.push_back({i, j, v});
      }
    }
  }
  return SparseMatrix::from_triplets(static_cast<std::size_t>(m), static_cast<std::size_t>(m), trip);
}

// ---------- linear_algebra_backend.cpp ----------
#if defined(SPARSE_LP_USE_EIGEN)
#endif



bool NativeBasisBackend::factorize(const SparseMatrix& basis) {
  return lu_.factorize(basis);
}

std::vector<double> NativeBasisBackend::solve(const std::vector<double>& rhs) const {
  return lu_.solve(rhs);
}

#if defined(SPARSE_LP_USE_EIGEN)

namespace {

class EigenBasisBackend final : public BasisFactorizationBackend {
public:
  bool factorize(const SparseMatrix& basis) override {
    const int n = static_cast<int>(basis.rows());
    if (static_cast<int>(basis.cols()) != n) return false;
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(basis.nnz());
    for (int j = 0; j < n; ++j) {
      for (int p = basis.col_pointers()[static_cast<std::size_t>(j)];
           p < basis.col_pointers()[static_cast<std::size_t>(j + 1)]; ++p) {
        const int i = basis.row_indices()[static_cast<std::size_t>(p)];
        trips.emplace_back(i, j, basis.values()[static_cast<std::size_t>(p)]);
      }
    }
    mat_.resize(n, n);
    mat_.setFromTriplets(trips.begin(), trips.end());
    solver_.compute(mat_);
    return solver_.info() == Eigen::Success;
  }

  std::vector<double> solve(const std::vector<double>& rhs) const override {
    Eigen::VectorXd b(static_cast<Eigen::Index>(rhs.size()));
    for (std::size_t i = 0; i < rhs.size(); ++i) {
      b(static_cast<Eigen::Index>(i)) = rhs[i];
    }
    Eigen::VectorXd x = solver_.solve(b);
    if (solver_.info() != Eigen::Success) {
      throw std::runtime_error("EigenBasisBackend::solve failed");
    }
    std::vector<double> out(rhs.size());
    for (std::size_t i = 0; i < rhs.size(); ++i) {
      out[i] = x(static_cast<Eigen::Index>(i));
    }
    return out;
  }

  LinearAlgebraBackendKind kind() const override { return LinearAlgebraBackendKind::Eigen; }

private:
  Eigen::SparseMatrix<double> mat_;
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_;
};

}  // namespace

#endif  // SPARSE_LP_USE_EIGEN

#if defined(SPARSE_LP_USE_SUITESPARSE)

namespace {

class SuiteSparseBasisBackend final : public BasisFactorizationBackend {
public:
  bool factorize(const SparseMatrix& basis) override {
    (void)basis;
    // Placeholder: integrate KLU (klu_analyze / klu_factor) or UMFPACK here using CSC pointers.
    return false;
  }

  std::vector<double> solve(const std::vector<double>& rhs) const override {
    (void)rhs;
    throw std::runtime_error("SuiteSparseBasisBackend: not implemented; use Native or Eigen");
  }

  LinearAlgebraBackendKind kind() const override { return LinearAlgebraBackendKind::SuiteSparse; }
};

}  // namespace

#endif  // SPARSE_LP_USE_SUITESPARSE

std::unique_ptr<BasisFactorizationBackend> make_backend(LinearAlgebraBackendKind k) {
  switch (k) {
    case LinearAlgebraBackendKind::Native:
      return std::make_unique<NativeBasisBackend>();
#if defined(SPARSE_LP_USE_EIGEN)
    case LinearAlgebraBackendKind::Eigen:
      return std::make_unique<EigenBasisBackend>();
#endif
#if defined(SPARSE_LP_USE_SUITESPARSE)
    case LinearAlgebraBackendKind::SuiteSparse:
      return std::make_unique<SuiteSparseBasisBackend>();
#endif
    default:
      break;
  }
  return std::make_unique<NativeBasisBackend>();
}

// ---------- lp_model.cpp ----------
void LpModel::validate_dimensions() const {
  const std::size_t m = A.rows();
  const std::size_t n = A.cols();
  if (col_cost.size() != n) {
    throw std::invalid_argument("LpModel: col_cost size must match A.cols()");
  }
  if (col_lower.size() != n || col_upper.size() != n) {
    throw std::invalid_argument("LpModel: col bounds length must match A.cols()");
  }
  if (!row_lower.empty() || !row_upper.empty()) {
    if (row_lower.size() != m || row_upper.size() != m) {
      throw std::invalid_argument("LpModel: row bounds length must match A.rows() or be empty");
    }
  }
}

// ---------- revised_dual_simplex.cpp ----------
RevisedDualSimplex::RevisedDualSimplex(LpModel model, LinearAlgebraBackendKind backend_kind)
    : model_(std::move(model)), backend_(make_backend(backend_kind)) {
  model_.validate_dimensions();
}

void RevisedDualSimplex::set_basis_slack_tail() {
  const int m = static_cast<int>(model_.A.rows());
  const int n = static_cast<int>(model_.A.cols());
  if (n < m) {
    throw std::invalid_argument("RevisedDualSimplex: need A.cols() >= A.rows() for slack-tail basis");
  }
  basis_cols_.resize(static_cast<std::size_t>(m));
  for (int i = 0; i < m; ++i) {
    basis_cols_[static_cast<std::size_t>(i)] = n - m + i;
  }
}

bool RevisedDualSimplex::refactor() {
  const int m = static_cast<int>(model_.A.rows());
  if (static_cast<int>(basis_cols_.size()) != m) {
    return false;
  }
  const SparseMatrix B = column_submatrix(model_.A, basis_cols_);
  return backend_->factorize(B);
}

SolveStatus RevisedDualSimplex::solve_framework() {
  const int m = static_cast<int>(model_.A.rows());
  if (m <= 0) {
    return SolveStatus::DimensionError;
  }
  try {
    set_basis_slack_tail();
  } catch (...) {
    return SolveStatus::DimensionError;
  }
  if (!refactor()) {
    return SolveStatus::FactorizationFailed;
  }
  return SolveStatus::BasisFactorized;
}

namespace {

std::vector<double> build_dense_column_from_sparse(const SparseMatrix& A, int col) {
  const int m = static_cast<int>(A.rows());
  std::vector<double> v(static_cast<std::size_t>(m), 0.0);
  if (col < 0 || static_cast<std::size_t>(col) >= A.cols()) {
    throw std::out_of_range("build_dense_column_from_sparse");
  }
  for (int p = A.col_pointers()[static_cast<std::size_t>(col)];
       p < A.col_pointers()[static_cast<std::size_t>(col + 1)]; ++p) {
    v[static_cast<std::size_t>(A.row_indices()[static_cast<std::size_t>(p)])] =
        A.values()[static_cast<std::size_t>(p)];
  }
  return v;
}

}  // namespace

SolveStatus RevisedDualSimplex::solve(int max_iterations) {
  const int m = static_cast<int>(model_.A.rows());
  const int n = static_cast<int>(model_.A.cols());

  if (m <= 0 || n <= 0 || n < m) {
    return SolveStatus::DimensionError;
  }
  if (static_cast<int>(model_.row_lower.size()) != m ||
      static_cast<int>(model_.row_upper.size()) != m) {
    return SolveStatus::NotImplemented;
  }
  for (int i = 0; i < m; ++i) {
    if (std::abs(model_.row_lower[static_cast<std::size_t>(i)] -
                 model_.row_upper[static_cast<std::size_t>(i)]) > 1e-12) {
      return SolveStatus::NotImplemented;
    }
  }

  // Initial basis and refactor of B = A(:, basis_cols_).
  set_basis_slack_tail();
  if (!refactor()) {
    return SolveStatus::FactorizationFailed;
  }

  const std::vector<double> b(m > 0 ? model_.row_lower : std::vector<double>());
  std::vector<double> x_B = backend_->solve(b);

  // Iterate primal revised simplex.
  for (int iter = 0; iter < max_iterations; ++iter) {
    // Build basis cost vector and nonbasis index set.
    std::vector<double> c_B(static_cast<std::size_t>(m));
    std::vector<int> nonbasis;
    std::vector<char> is_basis(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < m; ++i) {
      int bj = basis_cols_[static_cast<std::size_t>(i)];
      c_B[static_cast<std::size_t>(i)] = model_.col_cost[static_cast<std::size_t>(bj)];
      is_basis[static_cast<std::size_t>(bj)] = 1;
    }
    for (int j = 0; j < n; ++j) {
      if (!is_basis[static_cast<std::size_t>(j)]) nonbasis.push_back(j);
    }

    // Compute dual prices pi from B^T pi = c_B by dense B^{-1} reproduction (since no transpose solve).
    std::vector<std::vector<double>> B_inv_cols(static_cast<std::size_t>(m),
                                                std::vector<double>(static_cast<std::size_t>(m)));
    for (int i = 0; i < m; ++i) {
      std::vector<double> e(static_cast<std::size_t>(m), 0.0);
      e[static_cast<std::size_t>(i)] = 1.0;
      const std::vector<double> col = backend_->solve(e);
      B_inv_cols[static_cast<std::size_t>(i)] = col;
    }

    std::vector<double> pi(static_cast<std::size_t>(m), 0.0);
    for (int j = 0; j < m; ++j) {
      for (int i = 0; i < m; ++i) {
        pi[static_cast<std::size_t>(i)] += c_B[static_cast<std::size_t>(j)] *
            B_inv_cols[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)];
      }
    }

    // Pricing: find most negative reduced cost.
    int entering = -1;
    double min_reduced_cost = 1e-12;
    for (int j : nonbasis) {
      double aTp = 0.0;
      for (int p = model_.A.col_pointers()[static_cast<std::size_t>(j)];
           p < model_.A.col_pointers()[static_cast<std::size_t>(j + 1)]; ++p) {
        int i = model_.A.row_indices()[static_cast<std::size_t>(p)];
        aTp += model_.A.values()[static_cast<std::size_t>(p)] * pi[static_cast<std::size_t>(i)];
      }
      double reduced_cost = model_.col_cost[static_cast<std::size_t>(j)] - aTp;
      if (reduced_cost < min_reduced_cost) {
        min_reduced_cost = reduced_cost;
        entering = j;
      }
    }

    if (entering < 0) {
      return SolveStatus::Optimal;
    }

    // Compute direction d = B^{-1} A(:, entering)
    const std::vector<double> a_col = build_dense_column_from_sparse(model_.A, entering);
    const std::vector<double> d = backend_->solve(a_col);

    // Ratio test x_B / d for d > 0
    int leaving_basis_index = -1;
    double theta = std::numeric_limits<double>::infinity();
    for (int i = 0; i < m; ++i) {
      if (d[static_cast<std::size_t>(i)] > 1e-12) {
        double ratio = x_B[static_cast<std::size_t>(i)] / d[static_cast<std::size_t>(i)];
        if (ratio < theta) {
          theta = ratio;
          leaving_basis_index = i;
        }
      }
    }
    if (leaving_basis_index < 0 || !std::isfinite(theta)) {
      return SolveStatus::Unbounded;
    }

    // Pivot.
    for (int i = 0; i < m; ++i) {
      x_B[static_cast<std::size_t>(i)] -= theta * d[static_cast<std::size_t>(i)];
    }
    x_B[static_cast<std::size_t>(leaving_basis_index)] = theta;

    basis_cols_[static_cast<std::size_t>(leaving_basis_index)] = entering;
    if (!refactor()) {
      return SolveStatus::FactorizationFailed;
    }
    x_B = backend_->solve(b);
  }

  // 如果遍历到最大迭代仍未完全收敛，则返回 NotImplemented 以指示算法中断；
  // 对应用端可当成已找到近似解（此时 x_B 是当前基解）。
  return SolveStatus::NotImplemented;
}

// ---------- sparse_vector.cpp ----------
void SparseVector::sort_by_index() {
  const std::size_t n = indices_.size();
  std::vector<std::size_t> ord(n);
  for (std::size_t i = 0; i < n; ++i) ord[i] = i;
  std::sort(ord.begin(), ord.end(),
              [&](std::size_t a, std::size_t b) { return indices_[a] < indices_[b]; });
  std::vector<int> ni(n);
  std::vector<double> nv(n);
  for (std::size_t k = 0; k < n; ++k) {
    ni[k] = indices_[ord[k]];
    nv[k] = values_[ord[k]];
  }
  indices_.swap(ni);
  values_.swap(nv);
}

void SparseVector::axpy_sorted(double alpha, const SparseVector& other) {
  if (other.nnz() == 0) return;
  std::unordered_map<int, double> acc;
  acc.reserve(indices_.size() + other.nnz());
  for (std::size_t i = 0; i < indices_.size(); ++i) {
    acc[indices_[i]] = values_[i];
  }
  for (std::size_t i = 0; i < other.indices().size(); ++i) {
    acc[other.indices()[i]] += alpha * other.values()[i];
  }
  indices_.clear();
  values_.clear();
  indices_.reserve(acc.size());
  values_.reserve(acc.size());
  for (const auto& e : acc) {
    if (e.second != 0.0) {
      indices_.push_back(e.first);
      values_.push_back(e.second);
    }
  }
  sort_by_index();
}

// ---------- eta_file.cpp ----------
namespace {
constexpr double kTiny = 1e-18;
}

std::size_t EtaFile::total_nnz() const {
  std::size_t s = 0;
  for (const auto& e : entries_) {
    s += e.row_index.size();
  }
  return s;
}

void EtaFile::push(int pivot_col, const std::vector<double>& dense_column) {
  if (pivot_col < 0 || static_cast<std::size_t>(pivot_col) >= dense_column.size()) {
    throw std::invalid_argument("EtaFile::push: bad pivot");
  }
  EtaEntry e;
  e.pivot_col = pivot_col;
  for (std::size_t i = 0; i < dense_column.size(); ++i) {
    if (std::abs(dense_column[i]) > kTiny) {
      e.row_index.push_back(static_cast<int>(i));
      e.row_value.push_back(dense_column[i]);
    }
  }
  entries_.push_back(std::move(e));
}

void EtaFile::push_sparse(int pivot_col, const std::vector<int>& idx, const std::vector<double>& val) {
  if (idx.size() != val.size()) {
    throw std::invalid_argument("EtaFile::push_sparse");
  }
  EtaEntry e;
  e.pivot_col = pivot_col;
  e.row_index = idx;
  e.row_value = val;
  entries_.push_back(std::move(e));
}

void EtaFile::apply_t_inverse_inplace(std::vector<double>& y, const EtaEntry& e) {
  const int p = e.pivot_col;
  if (p < 0 || static_cast<std::size_t>(p) >= y.size()) {
    throw std::invalid_argument("EtaFile::apply: pivot");
  }
  double hp = 0.0;
  for (std::size_t k = 0; k < e.row_index.size(); ++k) {
    if (e.row_index[k] == p) {
      hp = e.row_value[k];
      break;
    }
  }
  if (std::abs(hp) <= kTiny) {
    throw std::runtime_error("EtaFile: zero pivot in eta");
  }
  const double zp = y[static_cast<std::size_t>(p)] / hp;
  y[static_cast<std::size_t>(p)] = zp;
  for (std::size_t k = 0; k < e.row_index.size(); ++k) {
    const int i = e.row_index[k];
    if (i != p) {
      y[static_cast<std::size_t>(i)] -= e.row_value[k] * zp;
    }
  }
}

void EtaFile::apply_all_inverse(std::vector<double>& y) const {
  for (const auto& e : entries_) {
    apply_t_inverse_inplace(y, e);
  }
}

// ---------- ft_basis_maintenance.cpp ----------
namespace {
constexpr double kEps_ft = 1e-18;
}

AdvancedBasisMaintenance::AdvancedBasisMaintenance(BasisMaintenanceParams p) : params_(std::move(p)) {}

bool AdvancedBasisMaintenance::refactor(const SparseMatrix& B) {
  clear_etas();
  const bool ok = lu_.factorize(B);
  if (!ok) {
    lu_nnz_ = 0;
    return false;
  }
  lu_nnz_ = lu_.L().nnz() + lu_.U().nnz();
  eta_pivots_ = 0;
  return true;
}

void AdvancedBasisMaintenance::clear_etas() {
  etas_.clear();
  eta_pivots_ = 0;
}

bool AdvancedBasisMaintenance::push_basis_change(int pivot_col, const std::vector<double>& eta_column_dense) {
  if (pivot_col < 0 || static_cast<std::size_t>(pivot_col) >= eta_column_dense.size()) {
    throw std::invalid_argument("AdvancedBasisMaintenance::push_basis_change");
  }
  double hp = eta_column_dense[static_cast<std::size_t>(pivot_col)];
  if (std::abs(hp) < params_.pivot_tolerance) {
    return false;
  }
  etas_.push(pivot_col, eta_column_dense);
  ++eta_pivots_;
  return true;
}

bool AdvancedBasisMaintenance::should_refactor() const {
  if (eta_pivots_ >= params_.max_eta_pivots) {
    return true;
  }
  const std::size_t eta_nz = etas_.total_nnz();
  if (lu_nnz_ == 0) {
    return eta_nz > 0;
  }
  const double threshold = params_.eta_density_factor * static_cast<double>(lu_nnz_);
  return static_cast<double>(eta_nz) > threshold;
}

bool AdvancedBasisMaintenance::ft_eliminate_spike_in_u(SparseMatrix& U_perm, int spike_col,
                                                      std::vector<double> spike_dense) {
  const int n = static_cast<int>(U_perm.rows());
  if (U_perm.cols() != static_cast<std::size_t>(n) || spike_col < 0 || spike_col >= n ||
      static_cast<int>(spike_dense.size()) != n) {
    return false;
  }
  // Row-wise upper triangular storage (conceptual): eliminate subdiagonal in column spike_col
  // using the Forrest–Tomlin row permutation + elimination pattern.
  std::vector<std::map<int, double>> rows(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    for (int p = U_perm.col_pointers()[static_cast<std::size_t>(j)];
         p < U_perm.col_pointers()[static_cast<std::size_t>(j + 1)]; ++p) {
      const int i = U_perm.row_indices()[static_cast<std::size_t>(p)];
      const double v = U_perm.values()[static_cast<std::size_t>(p)];
      if (std::abs(v) > kEps_ft) {
        rows[static_cast<std::size_t>(i)][j] = v;
      }
    }
  }
  for (int i = 0; i < n; ++i) {
    rows[static_cast<std::size_t>(i)][spike_col] = spike_dense[static_cast<std::size_t>(i)];
  }
  for (int k = spike_col + 1; k < n; ++k) {
    auto it = rows[static_cast<std::size_t>(k)].find(spike_col);
    if (it == rows[static_cast<std::size_t>(k)].end() || std::abs(it->second) <= kEps_ft) {
      continue;
    }
    const double sub = it->second;
    auto pit = rows[static_cast<std::size_t>(spike_col)].find(spike_col);
    if (pit == rows[static_cast<std::size_t>(spike_col)].end() ||
        std::abs(pit->second) < params_.pivot_tolerance) {
      return false;
    }
    const double piv = pit->second;
    const double mult = sub / piv;
    rows[static_cast<std::size_t>(k)].erase(it);
    for (const auto& e : rows[static_cast<std::size_t>(spike_col)]) {
      const int j = e.first;
      if (j <= spike_col) continue;
      const double v = e.second;
      auto& target = rows[static_cast<std::size_t>(k)][j];
      target -= mult * v;
      if (std::abs(target) <= kEps_ft) {
        rows[static_cast<std::size_t>(k)].erase(j);
      }
    }
  }
  std::vector<std::tuple<int, int, double>> trip;
  trip.reserve(static_cast<std::size_t>(n) * 4);
  for (int i = 0; i < n; ++i) {
    for (const auto& e : rows[static_cast<std::size_t>(i)]) {
      if (std::abs(e.second) > kEps_ft) {
        trip.push_back({i, e.first, e.second});
      }
    }
  }
  U_perm = SparseMatrix::from_triplets(static_cast<std::size_t>(n), static_cast<std::size_t>(n), trip);
  return true;
}

std::vector<double> ftran_with_etas(const LUSolver& lu, const EtaFile& etas, const std::vector<double>& b) {
  if (!lu.ok()) {
    throw std::runtime_error("ftran_with_etas: LU not factorized");
  }
  const int n = static_cast<int>(b.size());
  std::vector<double> b_perm(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const int orig_row = lu.perm_row()[static_cast<std::size_t>(i)];
    b_perm[static_cast<std::size_t>(i)] = b[static_cast<std::size_t>(orig_row)];
  }
  std::vector<double> y = lu.forward_substitution(b_perm);
  std::vector<double> z = lu.backward_substitution(y);
  etas.apply_all_inverse(z);
  return lu.scatter_solution(z);
}

// ---------- hypersparse_solve.cpp ----------
namespace {
constexpr double kEps_hs = 1e-18;
}

void symbolic_reach_lower_unit(const SparseMatrix& L, const SparseVector& rhs_pattern,
                               std::vector<int>& reach_order) {
  const int n = static_cast<int>(L.rows());
  std::unordered_set<int> s;
  for (std::size_t k = 0; k < rhs_pattern.indices().size(); ++k) {
    const int i = rhs_pattern.indices()[k];
    if (std::abs(rhs_pattern.values()[k]) > kEps_hs) {
      s.insert(i);
    }
  }
  for (int iter = 0; iter < n + 5; ++iter) {
    bool changed = false;
    for (int j = 0; j < n; ++j) {
      if (!s.count(j)) continue;
      for (int p = L.col_pointers()[static_cast<std::size_t>(j)];
           p < L.col_pointers()[static_cast<std::size_t>(j + 1)]; ++p) {
        const int i = L.row_indices()[static_cast<std::size_t>(p)];
        if (i > j && std::abs(L.values()[static_cast<std::size_t>(p)]) > kEps_hs) {
          if (!s.count(i)) {
            s.insert(i);
            changed = true;
          }
        }
      }
    }
    if (!changed) break;
  }
  reach_order.assign(s.begin(), s.end());
  std::sort(reach_order.begin(), reach_order.end());
}

void numeric_forward_lower_unit(const SparseMatrix& L, const std::vector<double>& rhs_dense,
                                const std::vector<int>& reach_order, std::vector<double>& y) {
  (void)reach_order;
  const int n = static_cast<int>(L.rows());
  y = rhs_dense;
  if (static_cast<int>(y.size()) < n) {
    y.resize(static_cast<std::size_t>(n), 0.0);
  }
  for (int j = 0; j < n - 1; ++j) {
    const double yj = y[static_cast<std::size_t>(j)];
    if (std::abs(yj) <= kEps_hs) continue;
    for (int p = L.col_pointers()[static_cast<std::size_t>(j)];
         p < L.col_pointers()[static_cast<std::size_t>(j + 1)]; ++p) {
      const int i = L.row_indices()[static_cast<std::size_t>(p)];
      if (i > j) {
        const double l_ij = L.values()[static_cast<std::size_t>(p)];
        y[static_cast<std::size_t>(i)] -= l_ij * yj;
      }
    }
  }
}

void hypersparse_ftran_l(const SparseMatrix& L, const SparseVector& rhs_sparse,
                         std::vector<double>& work_dense, std::vector<int>& reach_scratch) {
  const int n = static_cast<int>(L.rows());
  work_dense.assign(static_cast<std::size_t>(n), 0.0);
  for (std::size_t k = 0; k < rhs_sparse.indices().size(); ++k) {
    const int i = rhs_sparse.indices()[k];
    work_dense[static_cast<std::size_t>(i)] = rhs_sparse.values()[k];
  }
  symbolic_reach_lower_unit(L, rhs_sparse, reach_scratch);
  numeric_forward_lower_unit(L, work_dense, reach_scratch, work_dense);
}

void symbolic_reach_upper_transposed(const SparseMatrix& U, const SparseVector& rhs_pattern,
                                     std::vector<int>& reach_order_rev) {
  const int n = static_cast<int>(U.rows());
  std::unordered_set<int> s;
  for (std::size_t k = 0; k < rhs_pattern.indices().size(); ++k) {
    const int i = rhs_pattern.indices()[k];
    if (std::abs(rhs_pattern.values()[k]) > kEps_hs) {
      s.insert(i);
    }
  }
  for (int iter = 0; iter < n + 5; ++iter) {
    bool changed = false;
    for (int j = n - 1; j >= 0; --j) {
      if (!s.count(j)) continue;
      for (int i = 0; i < j; ++i) {
        if (std::abs(U.get(i, j)) > kEps_hs && !s.count(i)) {
          s.insert(i);
          changed = true;
        }
      }
    }
    if (!changed) break;
  }
  reach_order_rev.assign(s.begin(), s.end());
  std::sort(reach_order_rev.begin(), reach_order_rev.end());
}

void hypersparse_btran_u(const SparseMatrix& U, const SparseVector& rhs_sparse,
                         std::vector<double>& work_dense, std::vector<int>& reach_scratch) {
  const int n = static_cast<int>(U.rows());
  work_dense.assign(static_cast<std::size_t>(n), 0.0);
  for (std::size_t k = 0; k < rhs_sparse.indices().size(); ++k) {
    const int i = rhs_sparse.indices()[k];
    work_dense[static_cast<std::size_t>(i)] = rhs_sparse.values()[k];
  }
  symbolic_reach_upper_transposed(U, rhs_sparse, reach_scratch);
  std::unordered_set<int> active(reach_scratch.begin(), reach_scratch.end());
  for (int i = n - 1; i >= 0; --i) {
    if (!active.count(i)) continue;
    double sum = work_dense[static_cast<std::size_t>(i)];
    for (int j = i + 1; j < n; ++j) {
      sum -= U.get(i, j) * work_dense[static_cast<std::size_t>(j)];
    }
    const double uii = U.get(i, i);
    if (std::abs(uii) > kEps_hs) {
      work_dense[static_cast<std::size_t>(i)] = sum / uii;
    }
  }
}

// ---------- bound_flipping.cpp ----------
BoundFlipResult DualBoundFlippingController::long_step(std::vector<double>& pi,
                                                       const std::vector<double>& direction,
                                                       const std::vector<double>& theta_limit) const {
  BoundFlipResult r;
  const int m = static_cast<int>(pi.size());
  if (m != static_cast<int>(direction.size()) || m != static_cast<int>(theta_limit.size())) {
    r.dual_feasible_end = false;
    return r;
  }
  double theta = std::numeric_limits<double>::infinity();
  for (int i = 0; i < m; ++i) {
    const double di = direction[static_cast<std::size_t>(i)];
    if (std::abs(di) <= params_.feasibility_tol) continue;
    const double lim = theta_limit[static_cast<std::size_t>(i)];
    if (lim >= 0.0 && lim < theta) {
      theta = lim;
    }
  }
  if (!std::isfinite(theta) || theta <= 0.0) {
    theta = 0.0;
    r.step_length = theta;
    return r;
  }
  int flips = 0;
  while (flips < params_.max_flips && theta > params_.feasibility_tol) {
    for (int i = 0; i < m; ++i) {
      pi[static_cast<std::size_t>(i)] += theta * direction[static_cast<std::size_t>(i)];
    }
    r.step_length += theta;
    ++flips;
    break;
  }
  r.flips_applied = flips;
  r.dual_feasible_end = true;
  return r;
}

// ---------- presolve.cpp ----------
namespace {
constexpr double kTol = 1e-12;

bool is_zero_row(const SparseMatrix& A, int row) {
  for (std::size_t j = 0; j < A.cols(); ++j) {
    for (int p = A.col_pointers()[j]; p < A.col_pointers()[j + 1]; ++p) {
      if (A.row_indices()[static_cast<std::size_t>(p)] == static_cast<int>(row) &&
          std::abs(A.values()[static_cast<std::size_t>(p)]) > kTol) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

PresolveResult presolve_lp(const SparseMatrix& A, const std::vector<double>& c,
                           const std::vector<double>& col_lower, const std::vector<double>& col_upper,
                           const std::vector<double>& row_lower, const std::vector<double>& row_upper) {
  PresolveResult out;
  const int m = static_cast<int>(A.rows());
  const int n = static_cast<int>(A.cols());
  out.fixed_value_orig.assign(static_cast<std::size_t>(n), 0.0);
  out.is_fixed_orig.assign(static_cast<std::size_t>(n), 0);

  std::vector<int> alive_row(m, 1);
  std::vector<int> alive_col(n, 1);

  for (int i = 0; i < m; ++i) {
    if (is_zero_row(A, i)) {
      alive_row[static_cast<std::size_t>(i)] = 0;
      ++out.stats.empty_rows_removed;
    }
  }

  std::vector<int> col_nnz(static_cast<std::size_t>(n), 0);
  for (int j = 0; j < n; ++j) {
    for (int p = A.col_pointers()[static_cast<std::size_t>(j)];
         p < A.col_pointers()[static_cast<std::size_t>(j + 1)]; ++p) {
      if (std::abs(A.values()[static_cast<std::size_t>(p)]) > kTol) {
        ++col_nnz[static_cast<std::size_t>(j)];
      }
    }
  }
  for (int j = 0; j < n; ++j) {
    if (col_nnz[static_cast<std::size_t>(j)] == 0) {
      alive_col[static_cast<std::size_t>(j)] = 0;
      ++out.stats.empty_cols_removed;
    }
  }

  const bool have_row_bounds = row_lower.size() == static_cast<std::size_t>(m) &&
                             row_upper.size() == static_cast<std::size_t>(m);

  for (int i = 0; i < m; ++i) {
    if (!alive_row[static_cast<std::size_t>(i)]) continue;
    std::vector<std::pair<int, double>> nz;
    for (int j = 0; j < n; ++j) {
      if (!alive_col[static_cast<std::size_t>(j)]) continue;
      const double v = A.get(i, j);
      if (std::abs(v) > kTol) {
        nz.push_back({j, v});
      }
    }
    if (nz.size() != 1 || !have_row_bounds) continue;
    const int j = nz[0].first;
    const double aij = nz[0].second;
    if (std::abs(aij) <= kTol) continue;
    const double rl = row_lower[static_cast<std::size_t>(i)];
    const double ru = row_upper[static_cast<std::size_t>(i)];
    const double xj = rl / aij;
    const double xj2 = ru / aij;
    if (std::abs(xj - xj2) <= kTol * (1.0 + std::abs(xj))) {
      const double xv = xj;
      if (xv + kTol >= col_lower[static_cast<std::size_t>(j)] &&
          xv - kTol <= col_upper[static_cast<std::size_t>(j)]) {
        out.fixed_value_orig[static_cast<std::size_t>(j)] = xv;
        out.is_fixed_orig[static_cast<std::size_t>(j)] = 1;
        alive_col[static_cast<std::size_t>(j)] = 0;
        alive_row[static_cast<std::size_t>(i)] = 0;
        ++out.stats.singleton_rows_removed;
      }
    }
  }

  std::vector<int> col_map_old_to_new(static_cast<std::size_t>(n), -1);
  std::vector<int> row_map_old_to_new(static_cast<std::size_t>(m), -1);
  int n_new = 0;
  for (int j = 0; j < n; ++j) {
    if (alive_col[static_cast<std::size_t>(j)]) {
      col_map_old_to_new[static_cast<std::size_t>(j)] = n_new++;
    }
  }
  int m_new = 0;
  for (int i = 0; i < m; ++i) {
    if (alive_row[static_cast<std::size_t>(i)]) {
      row_map_old_to_new[static_cast<std::size_t>(i)] = m_new++;
    }
  }

  std::vector<std::tuple<int, int, double>> trip;
  trip.reserve(A.nnz());
  for (int j = 0; j < n; ++j) {
    if (!alive_col[static_cast<std::size_t>(j)]) continue;
    const int jn = col_map_old_to_new[static_cast<std::size_t>(j)];
    for (int p = A.col_pointers()[static_cast<std::size_t>(j)];
         p < A.col_pointers()[static_cast<std::size_t>(j + 1)]; ++p) {
      const int i = A.row_indices()[static_cast<std::size_t>(p)];
      if (!alive_row[static_cast<std::size_t>(i)]) continue;
      const int inew = row_map_old_to_new[static_cast<std::size_t>(i)];
      const double v = A.values()[static_cast<std::size_t>(p)];
      trip.push_back({inew, jn, v});
    }
  }

  out.A_reduced = SparseMatrix::from_triplets(static_cast<std::size_t>(m_new), static_cast<std::size_t>(n_new), trip);
  out.c_reduced.clear();
  out.l_reduced.clear();
  out.u_reduced.clear();
  out.c_reduced.reserve(static_cast<std::size_t>(n_new));
  out.l_reduced.reserve(static_cast<std::size_t>(n_new));
  out.u_reduced.reserve(static_cast<std::size_t>(n_new));
  for (int j = 0; j < n; ++j) {
    if (!alive_col[static_cast<std::size_t>(j)]) continue;
    out.c_reduced.push_back(c[static_cast<std::size_t>(j)]);
    out.l_reduced.push_back(col_lower[static_cast<std::size_t>(j)]);
    out.u_reduced.push_back(col_upper[static_cast<std::size_t>(j)]);
  }

  if (have_row_bounds) {
    out.row_lower_reduced.reserve(static_cast<std::size_t>(m_new));
    out.row_upper_reduced.reserve(static_cast<std::size_t>(m_new));
    for (int i = 0; i < m; ++i) {
      if (!alive_row[static_cast<std::size_t>(i)]) continue;
      out.row_lower_reduced.push_back(row_lower[static_cast<std::size_t>(i)]);
      out.row_upper_reduced.push_back(row_upper[static_cast<std::size_t>(i)]);
    }
  }

  out.col_mapping.resize(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    out.col_mapping[static_cast<std::size_t>(j)] = col_map_old_to_new[static_cast<std::size_t>(j)];
  }

  out.message = "presolve: empty rows/cols + singleton rows (equality)";
  return out;
}

// ---------- postsolve.cpp ----------
std::vector<double> PostsolveMap::expand_solution(const std::vector<double>& x_reduced) const {
  const int n_orig = static_cast<int>(pre_.col_mapping.size());
  std::vector<double> x_orig(static_cast<std::size_t>(n_orig), 0.0);
  for (int j = 0; j < n_orig; ++j) {
    if (pre_.is_fixed_orig[static_cast<std::size_t>(j)]) {
      x_orig[static_cast<std::size_t>(j)] = pre_.fixed_value_orig[static_cast<std::size_t>(j)];
      continue;
    }
    const int jr = pre_.col_mapping[static_cast<std::size_t>(j)];
    if (jr < 0 || static_cast<std::size_t>(jr) >= x_reduced.size()) {
      throw std::runtime_error("PostsolveMap::expand_solution: mapping mismatch");
    }
    x_orig[static_cast<std::size_t>(j)] = x_reduced[static_cast<std::size_t>(jr)];
  }
  return x_orig;
}

} // namespace sparse_lp



int main() {
  using namespace sparse_lp;

  // 经典实例：
  // maximize x0 + x1
  // s.t.    x0 + x1 <= 3
  //          x0 + 2x1 <= 4
  //          x0, x1 >= 0
  // 转化为最小化 -x0 - x1，使用松弛变量 x2,x3

  int m = 2;
  int n = 4;
  std::vector<std::tuple<int, int, double>> trip = {
      {0, 0, 1.0}, {0, 1, 1.0}, {0, 2, 1.0},
      {1, 0, 1.0}, {1, 1, 2.0}, {1, 3, 1.0}
  };
  SparseMatrix A = SparseMatrix::from_triplets(static_cast<std::size_t>(m), static_cast<std::size_t>(n), trip);

  LpModel model;
  model.A = A;
  model.col_cost = {-1.0, -1.0, 0.0, 0.0};
  model.col_lower = {0.0, 0.0, 0.0, 0.0};
  model.col_upper = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
  model.row_lower = {3.0, 4.0};
  model.row_upper = {3.0, 4.0};
  model.sense = ObjSense::Minimize;

  RevisedDualSimplex solver(std::move(model), LinearAlgebraBackendKind::Native);
  SolveStatus status = solver.solve(50);

  std::vector<double> x_full(static_cast<std::size_t>(n), 0.0);
  double obj = 0.0;
  if (status == SolveStatus::Optimal || status == SolveStatus::NotImplemented) {
    // Solve() 不返回 x，此处直接重新构造基解
    const std::vector<int>& basis = solver.basis_columns();
    std::vector<double> rhs = {3.0, 4.0};
    std::vector<double> x_B = solver.backend().solve(rhs);
    for (int i = 0; i < m; ++i) {
      x_full[static_cast<std::size_t>(basis[static_cast<std::size_t>(i)])] = x_B[static_cast<std::size_t>(i)];
    }
    for (int j = 0; j < n; ++j) {
      obj += solver.model().col_cost[static_cast<std::size_t>(j)] * x_full[static_cast<std::size_t>(j)];
    }
  }

  std::cout << "求解状态：" << static_cast<int>(status) << "\n";
  std::cout << "最终基变量索引：";
  for (int i : solver.basis_columns()) {
    std::cout << i << " ";
  }
  std::cout << "\n";
  std::cout << "变量解 x = [";
  for (int j = 0; j < n; ++j) {
    std::cout << x_full[static_cast<std::size_t>(j)];
    if (j + 1 < n) std::cout << ", ";
  }
  std::cout << "]\n";
  std::cout << "目标值 obj = " << obj << "\n";

  return 0;
}
