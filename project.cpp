// sparse_lp_complete.cpp — single-file bundle (Phases 1–5).
// Build:  c++ -std=c++17 -O2 sparse_lp_complete.cpp -o sparse_lp_test
// Run:    ./sparse_lp_test


#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include <filesystem>

#if defined(_WIN32)
#include <malloc.h>
#endif

#if defined(SPARSE_LP_USE_EIGEN)
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#endif

namespace sparse_lp {

static std::string trim(const std::string &s) {
  std::size_t first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  std::size_t last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

static std::vector<std::string> split_ws(const std::string &line) {
  std::vector<std::string> tokens;
  std::string token;
  for (char ch : line) {
    if (std::isspace(static_cast<unsigned char>(ch))) {
      if (!token.empty()) {
        tokens.push_back(token);
        token.clear();
      }
    } else {
      token.push_back(ch);
    }
  }
  if (!token.empty()) tokens.push_back(token);
  return tokens;
}

static double parse_double(const std::string &token) {
  try {
    return std::stod(token);
  } catch (...) {
    throw std::runtime_error("Cannot parse number: " + token);
  }
}

enum class MpsSection { None, Name, Rows, Columns, Rhs, Ranges, Bounds };

// Extract fixed-format MPS fields (columns are 1-indexed in the standard):
//   Field 1: cols 2-3   (e.g. row type letter, bound type)
//   Field 2: cols 5-12  (name)
//   Field 3: cols 15-22 (name)
//   Field 4: cols 25-36 (value)
//   Field 5: cols 40-47 (name)
//   Field 6: cols 50-61 (value)
// Fields are trimmed and may be empty. We only fall back to fixed-format when
// whitespace tokenization would be ambiguous (e.g. names with embedded spaces).
static std::string fixed_field(const std::string &line, std::size_t start, std::size_t len) {
  if (start >= line.size()) return std::string();
  std::size_t take = std::min(len, line.size() - start);
  std::string s = line.substr(start, take);
  // Trim spaces.
  std::size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

static void parse_mps(std::istream &in,
                      std::vector<std::string> &row_order,
                      std::vector<std::string> &col_order,
                      std::unordered_map<std::string, char> &row_type,
                      std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> &col_coeff,
                      std::unordered_map<std::string, double> &rhs_values,
                      std::unordered_map<std::string, double> &row_ranges,
                      std::unordered_map<std::string, double> &lower_bounds,
                      std::unordered_map<std::string, double> &upper_bounds,
                      std::string &objective_name) {
  MpsSection section = MpsSection::None;
  std::string active_rhs_set;
  std::string active_bounds_set;
  std::string line;

  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '*') continue;
    // Section headers begin in column 1 (no leading whitespace). Data lines are
    // indented. This distinction matters because RHS-set names like "RHS" can
    // legally appear as the first token of an indented data line and must NOT
    // be re-interpreted as the start of a new section.
    const bool is_header = !std::isspace(static_cast<unsigned char>(line[0]));
    // Drop trailing CR, but preserve internal spaces for fixed-format parsing.
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
    std::string trimmed = trim(line);
    if (trimmed.empty()) continue;
    if (is_header) {
      if (trimmed.rfind("NAME", 0) == 0) { section = MpsSection::Name; continue; }
      if (trimmed.rfind("ROWS", 0) == 0) { section = MpsSection::Rows; continue; }
      if (trimmed.rfind("COLUMNS", 0) == 0) { section = MpsSection::Columns; continue; }
      if (trimmed.rfind("RHS", 0) == 0) {
        section = MpsSection::Rhs;
        active_rhs_set.clear();
        auto htokens = split_ws(trimmed);
        if (htokens.size() >= 2) active_rhs_set = htokens[1];
        continue;
      }
      if (trimmed.rfind("RANGES", 0) == 0) { section = MpsSection::Ranges; continue; }
      if (trimmed.rfind("BOUNDS", 0) == 0) {
        section = MpsSection::Bounds;
        active_bounds_set.clear();
        continue;
      }
      if (trimmed.rfind("ENDATA", 0) == 0) break;
    }

    // Fixed-format fields (1-indexed in the standard, 0-indexed here):
    //   F1: 1-2, F2: 4-11, F3: 14-21, F4: 24-35, F5: 39-46, F6: 49-60.
    const std::string f1 = fixed_field(line, 1, 2);
    const std::string f2 = fixed_field(line, 4, 8);
    const std::string f3 = fixed_field(line, 14, 8);
    const std::string f4 = fixed_field(line, 24, 12);
    const std::string f5 = fixed_field(line, 39, 8);
    const std::string f6 = fixed_field(line, 49, 12);

    // Heuristic: a line is "fixed-format compatible" if the value fields, when
    // present, parse as numbers. Otherwise fall back to whitespace tokens.
    auto looks_like_number = [](const std::string &s) {
      if (s.empty()) return false;
      try { (void)std::stod(s); return true; } catch (...) { return false; }
    };

    auto tokens = split_ws(trimmed);
    if (tokens.empty()) continue;
    switch (section) {
      case MpsSection::Rows: {
        // Use fixed-format fields when the row name lives in F2 (cols 5-12),
        // because some MPS files have row names with embedded spaces.
        std::string type_str;
        std::string row_name;
        if (!f1.empty() && !f2.empty()) {
          type_str = f1;
          row_name = f2;
        } else {
          if (tokens.size() < 2) break;
          type_str = tokens[0];
          row_name = tokens[1];
        }
        if (type_str.empty()) break;
        char type = type_str[0];
        if (row_type.find(row_name) == row_type.end()) {
          row_order.push_back(row_name);
        }
        row_type[row_name] = type;
        if (type == 'N') {
          objective_name = row_name;
        }
        break;
      }
      case MpsSection::Columns: {
        // Prefer fixed-format if value fields parse as numbers; this correctly
        // handles names with embedded spaces (e.g. netlib/forplan "DEDO3 11").
        if (!f2.empty() && !f3.empty() && looks_like_number(f4)) {
          const std::string col_name = f2;
          if (col_coeff.find(col_name) == col_coeff.end()) {
            col_order.push_back(col_name);
          }
          col_coeff[col_name].emplace_back(f3, parse_double(f4));
          if (row_type.find(f3) == row_type.end()) row_order.push_back(f3);
          if (!f5.empty() && looks_like_number(f6)) {
            col_coeff[col_name].emplace_back(f5, parse_double(f6));
            if (row_type.find(f5) == row_type.end()) row_order.push_back(f5);
          }
          break;
        }
        if (tokens.size() < 3) break;
        std::string col_name = tokens[0];
        if (col_coeff.find(col_name) == col_coeff.end()) {
          col_order.push_back(col_name);
        }
        for (std::size_t i = 1; i + 1 < tokens.size(); i += 2) {
          std::string row_name = tokens[i];
          double value = parse_double(tokens[i + 1]);
          col_coeff[col_name].emplace_back(row_name, value);
          if (row_type.find(row_name) == row_type.end()) {
            row_order.push_back(row_name);
          }
        }
        break;
      }
      case MpsSection::Rhs: {
        if (!f2.empty() && !f3.empty() && looks_like_number(f4)) {
          // f2 is set name; f3/f4, optionally f5/f6 are row,value pairs.
          if (active_rhs_set.empty()) active_rhs_set = f2;
          rhs_values[f3] = parse_double(f4);
          if (!f5.empty() && looks_like_number(f6)) {
            rhs_values[f5] = parse_double(f6);
          }
          break;
        }
        if (tokens.size() < 2) break;
        std::size_t index = 0;
        if (row_type.find(tokens[0]) == row_type.end()) {
          if (active_rhs_set.empty()) active_rhs_set = tokens[0];
          index = 1;
        }
        while (index + 1 < tokens.size()) {
          std::string row_name = tokens[index];
          double value = parse_double(tokens[index + 1]);
          rhs_values[row_name] = value;
          index += 2;
        }
        break;
      }
      case MpsSection::Ranges: {
        if (!f2.empty() && !f3.empty() && looks_like_number(f4)) {
          row_ranges[f3] = parse_double(f4);
          if (!f5.empty() && looks_like_number(f6)) {
            row_ranges[f5] = parse_double(f6);
          }
          break;
        }
        if (tokens.size() < 2) break;
        std::size_t index = 0;
        if (row_type.find(tokens[0]) == row_type.end()) {
          index = 1;
        }
        while (index + 1 < tokens.size()) {
          std::string row_name = tokens[index];
          double value = parse_double(tokens[index + 1]);
          row_ranges[row_name] = value;
          index += 2;
        }
        break;
      }
      case MpsSection::Bounds: {
        // Fixed-format BOUNDS: F1=type, F2=set, F3=var, F4=value (optional).
        if (!f1.empty() && !f2.empty() && !f3.empty()) {
          const std::string &bound_type = f1;
          const std::string &bound_set = f2;
          const std::string &var_name = f3;
          double value = std::nan("0");
          if (!f4.empty() && looks_like_number(f4)) value = parse_double(f4);
          if (active_bounds_set.empty()) active_bounds_set = bound_set;
          if (bound_set != active_bounds_set) break;
          if (bound_type == "LO") lower_bounds[var_name] = value;
          else if (bound_type == "UP") upper_bounds[var_name] = value;
          else if (bound_type == "FX") { lower_bounds[var_name] = value; upper_bounds[var_name] = value; }
          else if (bound_type == "FR") {
            lower_bounds[var_name] = -std::numeric_limits<double>::infinity();
            upper_bounds[var_name] = std::numeric_limits<double>::infinity();
          } else if (bound_type == "MI") lower_bounds[var_name] = -std::numeric_limits<double>::infinity();
          else if (bound_type == "PL") upper_bounds[var_name] = std::numeric_limits<double>::infinity();
          else if (bound_type == "BV" || bound_type == "UI" || bound_type == "LI" || bound_type == "SC") {
            lower_bounds[var_name] = 0.0;
            upper_bounds[var_name] = 1.0;
          }
          break;
        }
        if (tokens.size() < 3) break;
        std::string bound_type = tokens[0];
        std::string bound_set = tokens[1];
        std::string var_name = tokens[2];
        double value = std::nan("0");
        if (tokens.size() >= 4) {
          value = parse_double(tokens[3]);
        }
        if (active_bounds_set.empty()) active_bounds_set = bound_set;
        if (bound_set != active_bounds_set) break;
        if (bound_type == "LO") {
          lower_bounds[var_name] = value;
        } else if (bound_type == "UP") {
          upper_bounds[var_name] = value;
        } else if (bound_type == "FX") {
          lower_bounds[var_name] = value;
          upper_bounds[var_name] = value;
        } else if (bound_type == "FR") {
          lower_bounds[var_name] = -std::numeric_limits<double>::infinity();
          upper_bounds[var_name] = std::numeric_limits<double>::infinity();
        } else if (bound_type == "MI") {
          lower_bounds[var_name] = -std::numeric_limits<double>::infinity();
        } else if (bound_type == "PL") {
          upper_bounds[var_name] = std::numeric_limits<double>::infinity();
        } else if (bound_type == "BV" || bound_type == "UI" || bound_type == "LI" || bound_type == "SC") {
          lower_bounds[var_name] = 0.0;
          upper_bounds[var_name] = 1.0;
        }
        break;
      }
      default:
        break;
    }
  }
}

static void effective_row_bounds(char type,
                                 double rhs_value,
                                 double range_value,
                                 double &lower,
                                 double &upper) {
  lower = -std::numeric_limits<double>::infinity();
  upper = std::numeric_limits<double>::infinity();
  switch (type) {
    case 'E':
      lower = upper = rhs_value;
      if (!std::isnan(range_value)) {
        if (range_value >= 0.0) {
          upper = rhs_value + range_value;
        } else {
          lower = rhs_value + range_value;
        }
      }
      break;
    case 'L':
      upper = rhs_value;
      if (!std::isnan(range_value)) {
        if (range_value >= 0.0) {
          lower = rhs_value - range_value;
        } else {
          lower = rhs_value;
          upper = rhs_value - range_value;
        }
      }
      break;
    case 'G':
      lower = rhs_value;
      if (!std::isnan(range_value)) {
        if (range_value >= 0.0) {
          upper = rhs_value + range_value;
        } else {
          lower = rhs_value + range_value;
          upper = rhs_value;
        }
      }
      break;
    default:
      break;
  }
}

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

  /// Solve A^T x = b (same permutations as factorize; b indexed in natural column order of A).
  std::vector<double> solve_transpose(const std::vector<double>& b) const;

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

  std::vector<double> upper_transpose_solve_perm(const std::vector<double>& c_perm) const;
  std::vector<double> lower_transpose_solve_perm(const std::vector<double>& w_perm) const;

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
  /// Solve basis^T x = rhs (dual / pricing multipliers).
  virtual std::vector<double> solve_transpose(const std::vector<double>& rhs) const = 0;
  virtual LinearAlgebraBackendKind kind() const = 0;
};

/// Native implementation backed by LUSolver.
class NativeBasisBackend final : public BasisFactorizationBackend {
public:
  bool factorize(const SparseMatrix& basis) override;
  std::vector<double> solve(const std::vector<double>& rhs) const override;
  std::vector<double> solve_transpose(const std::vector<double>& rhs) const override;
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
  NotImplemented,    ///< Full dual simplex iterations not wired yet.
  Infeasible         ///< Phase 1 detected primal infeasibility (sum of artificials > 0).
};

/// Revised dual simplex driver skeleton: maintains a basis B of m columns of A,
/// factors B with BasisFactorizationBackend, and exposes refactor hooks for pivots.
class RevisedDualSimplex {
public:
  explicit RevisedDualSimplex(LpModel model, LinearAlgebraBackendKind backend_kind,
                              bool allow_phase_one = true);

  const LpModel& model() const { return model_; }

  /// Use the last m columns of A as the initial basis (slack layout: A = [F | I]).
  void set_basis_slack_tail();

  /// Pick some consecutive m columns that yield a nonsingular basis (tries all offsets).
  bool set_basis_scan_consecutive();

  /// Inject an explicit initial basis (size must equal A.rows()). Caller is responsible for feasibility.
  bool set_initial_basis(std::vector<int> basis);

  /// Tag certain columns as "artificial" (Big-M priced). Once an artificial leaves
  /// the basis it is excluded from re-entering, mirroring the role of a Phase I
  /// auxiliary variable in textbook two-phase simplex.
  void set_is_artificial(std::vector<char> mask) { is_artificial_ = std::move(mask); }
  const std::vector<char>& is_artificial() const { return is_artificial_; }

  /// Refactorize current basis B = A(:, basis_cols_).
  bool refactor();

  SolveStatus solve_framework();
  /// Perform a simplified primal revised simplex iterations on standard slack-extended LP.
  SolveStatus solve(int max_iterations = 100);

  const std::vector<int>& basis_columns() const { return basis_cols_; }
  BasisFactorizationBackend& backend() { return *backend_; }
  const BasisFactorizationBackend& backend() const { return *backend_; }

private:
  bool phase_one(int max_iterations);

  LpModel model_;
  std::unique_ptr<BasisFactorizationBackend> backend_;
  std::vector<int> basis_cols_;
  std::vector<char> is_artificial_;  ///< Optional tag matching A.cols() length.
  bool allow_phase_one_{true};
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

/// IndexedVector: Sparse accumulator combining dense work array with sparse (index, value) tracking.
/// Enables efficient hypersparse operations (Gilbert-Peierls, weight updates) without O(n) scans.
/// Maintains: dense[i] for random access, sparse_indices for active indices, mark[i] for membership.
class IndexedVector {
public:
  explicit IndexedVector(std::size_t n) : dense_(n, 0.0), mark_(n, false) {
    sparse_indices_.reserve(n / 10);  // Expect sparse activity
  }

  /// Set dense[i] = v and mark index i as active if not already marked
  void scatter(int i, double v) {
    if (i < 0 || static_cast<std::size_t>(i) >= dense_.size()) return;
    const std::size_t ui = static_cast<std::size_t>(i);
    if (!mark_[ui]) {
      sparse_indices_.push_back(i);
      mark_[ui] = true;
    }
    dense_[ui] = v;
  }

  /// Fetch dense[i] (returns 0.0 if unmarked or out of range)
  double gather(int i) const {
    if (i < 0 || static_cast<std::size_t>(i) >= dense_.size()) return 0.0;
    const std::size_t ui = static_cast<std::size_t>(i);
    return mark_[ui] ? dense_[ui] : 0.0;
  }

  /// Retrieve all active (index, value) pairs where |value| > threshold
  std::vector<std::pair<int, double>> gather_all_nonzero(double threshold = 1e-20) const {
    std::vector<std::pair<int, double>> result;
    for (int i : sparse_indices_) {
      const std::size_t ui = static_cast<std::size_t>(i);
      if (std::abs(dense_[ui]) > threshold) {
        result.push_back({i, dense_[ui]});
      }
    }
    return result;
  }

  /// Retrieve sparse indices of all active entries (whether zero or not)
  const std::vector<int>& active_indices() const { return sparse_indices_; }

  /// Element access (dense vector)
  const std::vector<double>& dense() const { return dense_; }
  std::vector<double>& dense_mut() { return dense_; }

  /// Clear all marks and reset values
  void clear() {
    for (int i : sparse_indices_) {
      const std::size_t ui = static_cast<std::size_t>(i);
      mark_[ui] = false;
      dense_[ui] = 0.0;
    }
    sparse_indices_.clear();
  }

  std::size_t size() const { return dense_.size(); }

private:
  std::vector<double> dense_;
  std::vector<int> sparse_indices_;
  std::vector<bool> mark_;
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

  /// Goldfarb-Reid Recurrence Relation (Phase III):
  /// Update weights after a basis pivot using the FTRAN direction d and cross-product vector v.
  /// hat_w_i = max(1.0, w_i - 2*(d_i/d_p)*v_i + (d_i/d_p)^2*w_p) for i != p
  /// hat_w_p = w_p / d_p^2 for pivot row p
  /// Requires: gamma_[pivot_col] contains old w_p, and v = B^{-1} * rho_p^T (from extra FTRAN).
  void goldfarb_reid_update(int pivot_basis_index, int entering_col_index, 
                           double pivot_element, const std::vector<double>& ftran_direction,
                           const std::vector<double>& extra_ftran_v) {
    if (pivot_basis_index < 0 || entering_col_index < 0 ||
        static_cast<std::size_t>(entering_col_index) >= gamma_.size() ||
        ftran_direction.size() != extra_ftran_v.size()) {
      return;
    }

    if (std::abs(pivot_element) < 1e-16) return;
    
    const double w_p = gamma_[static_cast<std::size_t>(entering_col_index)];
    const double pivot_sq = pivot_element * pivot_element;
    
    // Update new pivot row weight
    gamma_[static_cast<std::size_t>(entering_col_index)] = 
        std::max(1.0, w_p / pivot_sq);
    
    // Update non-pivot rows using recurrence
    const int m = static_cast<int>(ftran_direction.size());
    for (int i = 0; i < m; ++i) {
      const double d_i = ftran_direction[static_cast<std::size_t>(i)];
      if (std::abs(d_i) < 1e-16) continue;
      
      // This weight corresponds to basis row i (old basis column before pivot)
      // Do NOT update the pivot row here (it's already updated above)
      if (i == pivot_basis_index) continue;
      
      const double v_i = extra_ftran_v[static_cast<std::size_t>(i)];
      const double ratio = d_i / pivot_element;
      const double w_i = gamma_[static_cast<std::size_t>(i)];
      
      const double new_w_i = w_i - 2.0 * ratio * v_i + ratio * ratio * w_p;
      gamma_[static_cast<std::size_t>(i)] = std::max(1.0, new_w_i);
    }
  }

private:
  std::vector<double> gamma_;
};

// ---------- ratio_test.hpp ----------
/// Harris Two-Pass Dual Ratio Test (CHUZC):
/// Selects entering column q that minimizes reduced cost while maximizing numerical stability.
/// Pass 1: Find maximum θ allowing slight dual feasibility relaxation (epsilon_d = 1e-7)
/// Pass 2: Among candidates satisfying θ, pick column with largest |α_pj| (most stable pivot)
struct HarrisRatioTestParams {
  double dual_feasibility_tolerance{1e-7};   ///< epsilon_d: allowed relative dual infeasibility
  double minimum_pivot_magnitude{1e-10};     ///< Minimum |alpha_pj| to consider a pivot
};

struct HarrisRatioTestResult {
  int entering_column{-1};       ///< The selected entering column q (-1 if ratio test fails)
  double theta_max{-1.0};        ///< Maximum step length from Pass 1
  double pivot_element{0.0};     ///< |α_pj| of the selected pivot
  int pass1_candidates{0};       ///< Count of columns satisfying Pass 1 threshold
};

/// Harris Two-Pass Ratio Test: Find entering column given leaving row.
/// Input: reduced_costs[j] = c_bar_j (reduced cost of column j)
///        alpha_p[j] = α_pj (pivot row j-th element)
/// Returns: selected entering column and associated step length
inline HarrisRatioTestResult harris_ratio_test(
    const std::vector<double>& reduced_costs,
    const std::vector<double>& alpha_p,
    const HarrisRatioTestParams& params = {}) {
  HarrisRatioTestResult result;
  
  if (reduced_costs.size() != alpha_p.size()) {
    return result;  // Dimension mismatch
  }
  
  const int n = static_cast<int>(reduced_costs.size());
  if (n <= 0) return result;
  
  const double eps_d = params.dual_feasibility_tolerance;
  const double min_pivot = params.minimum_pivot_magnitude;
  
  // PASS 1: Find maximum θ, allowing slight dual infeasibility
  // θ_max = min_j { (c_bar_j + eps_d) / |α_pj| : α_pj < 0 }
  result.theta_max = std::numeric_limits<double>::infinity();
  result.pass1_candidates = 0;
  
  for (int j = 0; j < n; ++j) {
    const double alpha_pj = alpha_p[j];
    if (alpha_pj < -min_pivot) {  // Only negative pivots are valid
      const double theta_j = (reduced_costs[j] + eps_d) / (-alpha_pj);
      if (theta_j < result.theta_max) {
        result.theta_max = theta_j;
      }
    }
  }
  
  if (!std::isfinite(result.theta_max) || result.theta_max < -min_pivot) {
    return result;  // No valid pivot; ratio test fails
  }
  
  // PASS 2: Find column with maximum |α_pj| among Pass 1 candidates
  // Candidates satisfy: (c_bar_j / |α_pj|) <= theta_max
  double best_pivot_magnitude = -1.0;
  
  for (int j = 0; j < n; ++j) {
    const double alpha_pj = alpha_p[j];
    if (alpha_pj < -min_pivot) {
      const double ratio_j = reduced_costs[j] / (-alpha_pj);
      
      // Within threshold (allowing for numerical tolerance)
      if (ratio_j <= result.theta_max + min_pivot) {
        const double mag_pj = std::abs(alpha_pj);
        if (mag_pj > best_pivot_magnitude) {
          best_pivot_magnitude = mag_pj;
          result.entering_column = j;
          result.pivot_element = alpha_pj;
          result.pass1_candidates++;
        }
      }
    }
  }
  
  return result;
}

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
  bool infeasible{false};
  bool unbounded{false};
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

  // Use more conservative epsilon for numerical stability
  const double eps = 1e-12;

  std::vector<std::map<int, double>> rows(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    for (int p = A.col_pointers()[static_cast<std::size_t>(j)];
         p < A.col_pointers()[static_cast<std::size_t>(j + 1)]; ++p) {
      const int i = A.row_indices()[static_cast<std::size_t>(p)];
      const double v = A.values()[static_cast<std::size_t>(p)];
      if (std::abs(v) > eps) {
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
      // Try to find a better pivot in the same column
      double best_pivot_val = 0.0;
      int best_pivot_row = -1;
      for (int i = k; i < n; ++i) {
        const auto& row = rows[static_cast<std::size_t>(i)];
        auto it = row.find(k);
        if (it != row.end()) {
          double val = std::abs(it->second);
          if (val > best_pivot_val) {
            best_pivot_val = val;
            best_pivot_row = i;
          }
        }
      }
      if (best_pivot_row >= 0 && best_pivot_val > kEps) {
        // Swap rows to bring better pivot to diagonal
        swap_rows(rows, k, best_pivot_row);
        std::swap(perm_r_[static_cast<std::size_t>(k)], perm_r_[static_cast<std::size_t>(best_pivot_row)]);
        pit = pivot_row.find(k);
      } else {
        return false;  // No suitable pivot found
      }
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

std::vector<double> LUSolver::upper_transpose_solve_perm(const std::vector<double>& c_perm) const {
  const int n = n_;
  std::vector<double> w(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    double sum = c_perm[static_cast<std::size_t>(i)];
    for (int p = U_.col_pointers()[static_cast<std::size_t>(i)];
         p < U_.col_pointers()[static_cast<std::size_t>(i + 1)]; ++p) {
      const int k = U_.row_indices()[static_cast<std::size_t>(p)];
      const double v = U_.values()[static_cast<std::size_t>(p)];
      if (k < i) {
        sum -= v * w[static_cast<std::size_t>(k)];
      }
    }
    const double uii = U_.get(i, i);
    if (std::abs(uii) <= 1e-14) {
      throw std::runtime_error("LUSolver::upper_transpose_solve_perm: singular pivot");
    }
    w[static_cast<std::size_t>(i)] = sum / uii;
  }
  return w;
}

std::vector<double> LUSolver::lower_transpose_solve_perm(const std::vector<double>& w_perm) const {
  const int n = n_;
  std::vector<double> v(static_cast<std::size_t>(n), 0.0);
  for (int j = n - 1; j >= 0; --j) {
    double sum = w_perm[static_cast<std::size_t>(j)];
    for (int p = L_.col_pointers()[static_cast<std::size_t>(j)];
         p < L_.col_pointers()[static_cast<std::size_t>(j + 1)]; ++p) {
      const int i = L_.row_indices()[static_cast<std::size_t>(p)];
      const double l_ij = L_.values()[static_cast<std::size_t>(p)];
      if (i > j) {
        sum -= l_ij * v[static_cast<std::size_t>(i)];
      }
    }
    v[static_cast<std::size_t>(j)] = sum;
  }
  return v;
}

std::vector<double> LUSolver::solve_transpose(const std::vector<double>& c_natural) const {
  if (!factor_ok_) {
    throw std::runtime_error("LUSolver: factorize before solve_transpose");
  }
  if (static_cast<int>(c_natural.size()) != n_) {
    throw std::runtime_error("LUSolver::solve_transpose: rhs dimension mismatch");
  }
  std::vector<double> c_perm(static_cast<std::size_t>(n_));
  for (int j = 0; j < n_; ++j) {
    const int orig_col = perm_c_[static_cast<std::size_t>(j)];
    c_perm[static_cast<std::size_t>(j)] = c_natural[static_cast<std::size_t>(orig_col)];
  }
  std::vector<double> w = upper_transpose_solve_perm(c_perm);
  std::vector<double> v_perm = lower_transpose_solve_perm(w);
  std::vector<double> pi(static_cast<std::size_t>(n_), 0.0);
  for (int i = 0; i < n_; ++i) {
    const int orig_row = perm_r_[static_cast<std::size_t>(i)];
    pi[static_cast<std::size_t>(orig_row)] = v_perm[static_cast<std::size_t>(i)];
  }
  return pi;
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
  lu_.set_column_threshold(0.0);
  return lu_.factorize(basis);
}

std::vector<double> NativeBasisBackend::solve(const std::vector<double>& rhs) const {
  return lu_.solve(rhs);
}

std::vector<double> NativeBasisBackend::solve_transpose(const std::vector<double>& rhs) const {
  return lu_.solve_transpose(rhs);
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

  std::vector<double> solve_transpose(const std::vector<double>& rhs) const override {
    Eigen::VectorXd b(static_cast<Eigen::Index>(rhs.size()));
    for (std::size_t i = 0; i < rhs.size(); ++i) {
      b(static_cast<Eigen::Index>(i)) = rhs[i];
    }
    Eigen::VectorXd x = solver_.transpose().solve(b);
    if (solver_.info() != Eigen::Success) {
      throw std::runtime_error("EigenBasisBackend::solve_transpose failed");
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

  std::vector<double> solve_transpose(const std::vector<double>& rhs) const override {
    (void)rhs;
    throw std::runtime_error("SuiteSparseBasisBackend: not implemented; use Native or Eigen");
  }

  LinearAlgebraBackendKind kind() const override { return LinearAlgebraBackendKind::SuiteSparse; }
};

}  // namespace

#endif  // SPARSE_LP_USE_SUITESPARSE

// Robust dense LU backend with partial pivoting. The sparse Markowitz LU above
// is needed for very large bases, but for the LPs we exercise here a dense
// O(m^3) factorization is both cheap and numerically reliable -- crucial for
// the simplex driver, where a failed refactor sends us into a recovery path
// that can break Bland's anti-cycling guarantees.
class DenseLUBackend final : public BasisFactorizationBackend {
public:
  bool factorize(const SparseMatrix& basis) override {
    const int n = static_cast<int>(basis.rows());
    if (static_cast<int>(basis.cols()) != n || n == 0) return false;
    n_ = n;
    A_.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    perm_.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) perm_[static_cast<std::size_t>(i)] = i;
    // Scatter CSC into a row-major dense buffer.
    for (int j = 0; j < n; ++j) {
      for (int p = basis.col_pointers()[static_cast<std::size_t>(j)];
           p < basis.col_pointers()[static_cast<std::size_t>(j + 1)]; ++p) {
        const int i = basis.row_indices()[static_cast<std::size_t>(p)];
        A_[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)] +=
            basis.values()[static_cast<std::size_t>(p)];
      }
    }
    constexpr double kSingTol = 1e-14;
    for (int k = 0; k < n; ++k) {
      // Partial pivoting: find row with largest |A[i,k]| for i >= k.
      int piv = k;
      double mx = std::abs(A_[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) +
                              static_cast<std::size_t>(k)]);
      for (int i = k + 1; i < n; ++i) {
        const double v = std::abs(A_[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                                     static_cast<std::size_t>(k)]);
        if (v > mx) { mx = v; piv = i; }
      }
      if (mx < kSingTol) return false;
      if (piv != k) {
        for (int j = 0; j < n; ++j) {
          std::swap(
              A_[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)],
              A_[static_cast<std::size_t>(piv) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)]);
        }
        std::swap(perm_[static_cast<std::size_t>(k)], perm_[static_cast<std::size_t>(piv)]);
      }
      const double pivot = A_[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) +
                              static_cast<std::size_t>(k)];
      for (int i = k + 1; i < n; ++i) {
        double& a_ik = A_[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
                          static_cast<std::size_t>(k)];
        if (a_ik == 0.0) continue;
        const double mult = a_ik / pivot;
        a_ik = mult;  // Store L[i,k]
        for (int j = k + 1; j < n; ++j) {
          A_[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)] -=
              mult * A_[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)];
        }
      }
    }
    factor_ok_ = true;
    return true;
  }

  std::vector<double> solve(const std::vector<double>& rhs) const override {
    if (!factor_ok_) throw std::runtime_error("DenseLUBackend: factorize before solve");
    const int n = n_;
    if (static_cast<int>(rhs.size()) != n) {
      throw std::runtime_error("DenseLUBackend::solve dim mismatch");
    }
    std::vector<double> y(static_cast<std::size_t>(n));
    // Apply row permutation: y = P * rhs.
    for (int i = 0; i < n; ++i) {
      y[static_cast<std::size_t>(i)] = rhs[static_cast<std::size_t>(perm_[static_cast<std::size_t>(i)])];
    }
    // Forward substitution L y = Pb (L unit-diagonal, stored in strictly lower part).
    for (int i = 0; i < n; ++i) {
      double s = y[static_cast<std::size_t>(i)];
      for (int j = 0; j < i; ++j) {
        s -= A_[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)] *
             y[static_cast<std::size_t>(j)];
      }
      y[static_cast<std::size_t>(i)] = s;
    }
    // Backward substitution U x = y.
    std::vector<double> x(static_cast<std::size_t>(n));
    for (int i = n - 1; i >= 0; --i) {
      double s = y[static_cast<std::size_t>(i)];
      for (int j = i + 1; j < n; ++j) {
        s -= A_[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)] *
             x[static_cast<std::size_t>(j)];
      }
      x[static_cast<std::size_t>(i)] = s /
          A_[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)];
    }
    return x;
  }

  std::vector<double> solve_transpose(const std::vector<double>& rhs) const override {
    if (!factor_ok_) throw std::runtime_error("DenseLUBackend: factorize before solve_transpose");
    const int n = n_;
    if (static_cast<int>(rhs.size()) != n) {
      throw std::runtime_error("DenseLUBackend::solve_transpose dim mismatch");
    }
    // Solve U^T y = rhs (i.e., upper-triangular system with U^T which is lower).
    std::vector<double> y(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      double s = rhs[static_cast<std::size_t>(i)];
      for (int j = 0; j < i; ++j) {
        s -= A_[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)] *
             y[static_cast<std::size_t>(j)];
      }
      y[static_cast<std::size_t>(i)] = s /
          A_[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)];
    }
    // Solve L^T z = y (L unit-diagonal stored in strictly lower part).
    std::vector<double> z(static_cast<std::size_t>(n));
    for (int i = n - 1; i >= 0; --i) {
      double s = y[static_cast<std::size_t>(i)];
      for (int j = i + 1; j < n; ++j) {
        s -= A_[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)] *
             z[static_cast<std::size_t>(j)];
      }
      z[static_cast<std::size_t>(i)] = s;
    }
    // Apply P^T: out[perm[i]] = z[i].
    std::vector<double> out(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      out[static_cast<std::size_t>(perm_[static_cast<std::size_t>(i)])] = z[static_cast<std::size_t>(i)];
    }
    return out;
  }

  LinearAlgebraBackendKind kind() const override { return LinearAlgebraBackendKind::Native; }

private:
  int n_{0};
  bool factor_ok_{false};
  std::vector<double> A_;        // Row-major; L (strict lower, unit diag) and U (upper) overwritten.
  std::vector<int> perm_;        // Row permutation P[k] = perm_[k].
};

std::unique_ptr<BasisFactorizationBackend> make_backend(LinearAlgebraBackendKind k) {
  switch (k) {
    case LinearAlgebraBackendKind::Native:
      return std::make_unique<DenseLUBackend>();
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
  return std::make_unique<DenseLUBackend>();
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

/// Converts LP problem to standard form: min c^T x s.t. A x = b, x >= 0
/// Handles variable bounds and range constraints by introducing slack/surplus variables
struct StandardFormLp {
  SparseMatrix A;
  std::vector<double> c;
  std::vector<double> b;
  std::vector<int> original_col_to_std;
  std::unordered_map<int, double> slack_bounds;
};

static StandardFormLp convert_to_standard_form(const LpModel& model) {
  StandardFormLp result;
  const int orig_n = static_cast<int>(model.A.cols());
  const int m = static_cast<int>(model.A.rows());
  
  std::vector<std::tuple<int, int, double>> triplets;
  std::vector<double> new_costs;
  result.original_col_to_std.assign(orig_n, -1);
  
  int new_col = 0;
  
  // Add variables for original columns with bounds
  for (int j = 0; j < orig_n; ++j) {
    double lb = model.col_lower[static_cast<std::size_t>(j)];
    double ub = model.col_upper[static_cast<std::size_t>(j)];
    
    // Handle free variables and non-zero lower bounds
    if (lb == -std::numeric_limits<double>::infinity() && 
        ub == std::numeric_limits<double>::infinity()) {
      // Free variable: split into x+ - x-
      result.original_col_to_std[static_cast<std::size_t>(j)] = new_col;
      for (int p = model.A.col_pointers()[static_cast<std::size_t>(j)];
           p < model.A.col_pointers()[static_cast<std::size_t>(j + 1)]; ++p) {
        int i = model.A.row_indices()[static_cast<std::size_t>(p)];
        double v = model.A.values()[static_cast<std::size_t>(p)];
        triplets.push_back({i, new_col, v});
        triplets.push_back({i, new_col + 1, -v});
      }
      new_costs.push_back(model.col_cost[static_cast<std::size_t>(j)]);
      new_costs.push_back(-model.col_cost[static_cast<std::size_t>(j)]);
      new_col += 2;
    } else {
      // Bounded variable: shift if necessary
      double shift = (lb > -std::numeric_limits<double>::infinity()) ? lb : 0.0;
      result.original_col_to_std[static_cast<std::size_t>(j)] = new_col;
      
      for (int p = model.A.col_pointers()[static_cast<std::size_t>(j)];
           p < model.A.col_pointers()[static_cast<std::size_t>(j + 1)]; ++p) {
        int i = model.A.row_indices()[static_cast<std::size_t>(p)];
        double v = model.A.values()[static_cast<std::size_t>(p)];
        triplets.push_back({i, new_col, v});
      }
      new_costs.push_back(model.col_cost[static_cast<std::size_t>(j)]);
      
      // Store bound info
      if (ub < std::numeric_limits<double>::infinity()) {
        result.slack_bounds[new_col] = ub - shift;
      }
      new_col++;
    }
  }
  
  // Add slack variables for row constraints
  for (int i = 0; i < m; ++i) {
    double row_lb = model.row_lower[static_cast<std::size_t>(i)];
    double row_ub = model.row_upper[static_cast<std::size_t>(i)];
    
    if (row_lb > -std::numeric_limits<double>::infinity() && 
        row_ub < std::numeric_limits<double>::infinity() &&
        std::abs(row_lb - row_ub) > 1e-12) {
      // Inequality constraint: add slack variable
      triplets.push_back({i, new_col, 1.0});
      new_costs.push_back(0.0);
      new_col++;
    }
  }
  
  result.A = SparseMatrix::from_triplets(static_cast<std::size_t>(m), 
                                         static_cast<std::size_t>(new_col), triplets);
  result.c = new_costs;
  
  // Build RHS vector
  result.b.assign(m, 0.0);
  for (int i = 0; i < m; ++i) {
    result.b[static_cast<std::size_t>(i)] = model.row_lower[static_cast<std::size_t>(i)];
  }
  
  return result;
}

/// Result of converting an arbitrary LP to canonical form
///   min c^T y   s.t.  A y = b,  y >= 0,  b >= 0
/// where the LAST m_can columns of A form an identity matrix and serve as the
/// initial primal-feasible basis (slacks for naturally-feasible rows, artificial
/// variables priced with Big-M for the rest). Variable shifts and free-variable
/// splits are recorded so the canonical objective offset can be added back.
struct CanonicalForm {
  LpModel model;
  std::vector<int> initial_basis;
  std::vector<char> is_artificial;
  double big_m{0.0};
  double obj_offset{0.0};
  int n_orig{0};
};

static CanonicalForm to_canonical_form(const LpModel& in) {
  CanonicalForm cf;
  cf.n_orig = static_cast<int>(in.A.cols());
  const int m_orig = static_cast<int>(in.A.rows());
  const int n_orig = cf.n_orig;

  // Stage 1: variable substitution.
  // For each original column j we emit one or two y columns and a shift so
  //   x_orig[j] = sign_pos * y_pos[j] + sign_neg * y_neg[j] + shift[j]
  // where each y_* >= 0 and at most one of y_neg is present (free split).
  struct VarMap {
    int y_pos{-1};
    int y_neg{-1};
    double shift{0.0};
    int sign_pos{1};
    double upper{std::numeric_limits<double>::infinity()};
  };
  std::vector<VarMap> vmap(static_cast<std::size_t>(n_orig));
  std::vector<double> y_cost;
  std::vector<int> y_origin_col;
  y_cost.reserve(static_cast<std::size_t>(n_orig));
  y_origin_col.reserve(static_cast<std::size_t>(n_orig));

  for (int j = 0; j < n_orig; ++j) {
    const double lb = in.col_lower[static_cast<std::size_t>(j)];
    const double ub = in.col_upper[static_cast<std::size_t>(j)];
    const double cj = in.col_cost[static_cast<std::size_t>(j)];
    VarMap vm;
    const bool lb_fin = std::isfinite(lb);
    const bool ub_fin = std::isfinite(ub);
    if (lb_fin && ub_fin) {
      vm.shift = lb;
      vm.sign_pos = 1;
      vm.upper = std::max(0.0, ub - lb);
      vm.y_pos = static_cast<int>(y_cost.size());
      y_cost.push_back(cj);
      y_origin_col.push_back(j);
      cf.obj_offset += cj * lb;
    } else if (lb_fin && !ub_fin) {
      vm.shift = lb;
      vm.sign_pos = 1;
      vm.upper = std::numeric_limits<double>::infinity();
      vm.y_pos = static_cast<int>(y_cost.size());
      y_cost.push_back(cj);
      y_origin_col.push_back(j);
      cf.obj_offset += cj * lb;
    } else if (!lb_fin && ub_fin) {
      vm.shift = ub;
      vm.sign_pos = -1;
      vm.upper = std::numeric_limits<double>::infinity();
      vm.y_pos = static_cast<int>(y_cost.size());
      y_cost.push_back(-cj);
      y_origin_col.push_back(j);
      cf.obj_offset += cj * ub;
    } else {
      vm.shift = 0.0;
      vm.sign_pos = 1;
      vm.upper = std::numeric_limits<double>::infinity();
      vm.y_pos = static_cast<int>(y_cost.size());
      y_cost.push_back(cj);
      y_origin_col.push_back(j);
      vm.y_neg = static_cast<int>(y_cost.size());
      y_cost.push_back(-cj);
      y_origin_col.push_back(j);
    }
    vmap[static_cast<std::size_t>(j)] = vm;
  }

  const int n_y = static_cast<int>(y_cost.size());

  // Stage 2: emit constraint triplets in y-space.
  // Each original row i produces 1 (E / one-sided) or 2 (ranged) canonical rows.
  // Track per-canonical-row: rhs_eff, sign (+1 for <= form, -1 for >= which we flip
  // by negating the row), and whether the row is structural equality.
  struct RowDesc {
    enum class Kind { LE, GE, EQ };
    Kind kind;
    double rhs;  // After variable shifts; before final sign flip.
  };

  std::vector<RowDesc> rdesc;
  rdesc.reserve(static_cast<std::size_t>(m_orig) * 2);
  std::vector<std::tuple<int, int, double>> trip;
  trip.reserve(in.A.nnz() * 2 + static_cast<std::size_t>(m_orig));

  auto emit_row_from = [&](int orig_row, int new_row, double shift_rhs_accum) {
    for (int p = in.A.col_pointers()[static_cast<std::size_t>(orig_row)];
         p < in.A.col_pointers()[static_cast<std::size_t>(orig_row + 1)]; ++p) {
      // CSC: col_pointers index columns. We need row iteration; we iterate cols below instead.
      (void)p; (void)new_row; (void)shift_rhs_accum;
    }
  };
  (void)emit_row_from;

  // Compute per-row shift contribution from variable shift: rhs_shift[i] = sum_j A[i,j] * shift_j
  std::vector<double> rhs_shift(static_cast<std::size_t>(m_orig), 0.0);
  for (int j = 0; j < n_orig; ++j) {
    const double sh = vmap[static_cast<std::size_t>(j)].shift;
    if (sh == 0.0) continue;
    for (int p = in.A.col_pointers()[static_cast<std::size_t>(j)];
         p < in.A.col_pointers()[static_cast<std::size_t>(j + 1)]; ++p) {
      const int i = in.A.row_indices()[static_cast<std::size_t>(p)];
      const double v = in.A.values()[static_cast<std::size_t>(p)];
      rhs_shift[static_cast<std::size_t>(i)] += v * sh;
    }
  }

  // For each original column j, scatter A[:,j] into the y-space row entries
  // (translated by sign_pos and possibly the negative split column).
  // We accumulate into a working row->col->value sparse map per "instance" of canonical row.
  // Since each original row produces at most 2 canonical rows, we build the canonical
  // row indices first, then scatter once per (canonical row, original column).
  std::vector<int> orig_row_to_canon_first(static_cast<std::size_t>(m_orig), -1);
  std::vector<int> orig_row_to_canon_second(static_cast<std::size_t>(m_orig), -1);

  for (int i = 0; i < m_orig; ++i) {
    const double rl = in.row_lower[static_cast<std::size_t>(i)];
    const double ru = in.row_upper[static_cast<std::size_t>(i)];
    const bool rl_fin = std::isfinite(rl);
    const bool ru_fin = std::isfinite(ru);
    const double sh = rhs_shift[static_cast<std::size_t>(i)];
    if (!rl_fin && !ru_fin) {
      // Free row: drop.
      continue;
    }
    if (rl_fin && ru_fin) {
      const double tol = 1e-9 * (1.0 + std::max(std::abs(rl), std::abs(ru)));
      if (std::abs(rl - ru) <= tol) {
        orig_row_to_canon_first[static_cast<std::size_t>(i)] = static_cast<int>(rdesc.size());
        rdesc.push_back({RowDesc::Kind::EQ, rl - sh});
      } else {
        orig_row_to_canon_first[static_cast<std::size_t>(i)] = static_cast<int>(rdesc.size());
        rdesc.push_back({RowDesc::Kind::LE, ru - sh});
        orig_row_to_canon_second[static_cast<std::size_t>(i)] = static_cast<int>(rdesc.size());
        rdesc.push_back({RowDesc::Kind::GE, rl - sh});
      }
    } else if (ru_fin) {
      orig_row_to_canon_first[static_cast<std::size_t>(i)] = static_cast<int>(rdesc.size());
      rdesc.push_back({RowDesc::Kind::LE, ru - sh});
    } else {
      orig_row_to_canon_first[static_cast<std::size_t>(i)] = static_cast<int>(rdesc.size());
      rdesc.push_back({RowDesc::Kind::GE, rl - sh});
    }
  }

  auto push_y_entries = [&](int canon_row, int orig_col, double coef) {
    if (coef == 0.0) return;
    const VarMap& vm = vmap[static_cast<std::size_t>(orig_col)];
    if (vm.y_pos >= 0) {
      const double v_pos = (vm.sign_pos > 0 ? coef : -coef);
      trip.emplace_back(canon_row, vm.y_pos, v_pos);
    }
    if (vm.y_neg >= 0) {
      trip.emplace_back(canon_row, vm.y_neg, -coef);
    }
  };

  for (int j = 0; j < n_orig; ++j) {
    for (int p = in.A.col_pointers()[static_cast<std::size_t>(j)];
         p < in.A.col_pointers()[static_cast<std::size_t>(j + 1)]; ++p) {
      const int i = in.A.row_indices()[static_cast<std::size_t>(p)];
      const double v = in.A.values()[static_cast<std::size_t>(p)];
      const int r1 = orig_row_to_canon_first[static_cast<std::size_t>(i)];
      const int r2 = orig_row_to_canon_second[static_cast<std::size_t>(i)];
      if (r1 >= 0) push_y_entries(r1, j, v);
      if (r2 >= 0) push_y_entries(r2, j, v);
    }
  }

  // Stage 3: add explicit upper bound rows for shifted variables (y_pos <= upper).
  for (int j = 0; j < n_orig; ++j) {
    const VarMap& vm = vmap[static_cast<std::size_t>(j)];
    if (vm.y_pos < 0) continue;
    if (!std::isfinite(vm.upper)) continue;
    const int rrow = static_cast<int>(rdesc.size());
    rdesc.push_back({RowDesc::Kind::LE, vm.upper});
    trip.emplace_back(rrow, vm.y_pos, 1.0);
  }

  const int m_can = static_cast<int>(rdesc.size());

  // Stage 4: for each canonical row, possibly flip sign so that the slack/artificial
  // basis variable is +1 with feasible value (b >= 0). Then append slack/surplus and
  // artificial columns as needed.
  // We will collect:
  //   - aux_trip: triplets for slack/surplus/artificial columns (positions filled later)
  //   - basis_col_per_row[i]: column index serving as initial basis for row i
  //   - aux_cost: cost vector for aux columns
  //   - aux_is_artif: whether each aux column is an artificial (Big-M priced)
  std::vector<double> b_can(static_cast<std::size_t>(m_can), 0.0);
  std::vector<char> row_negate(static_cast<std::size_t>(m_can), 0);
  std::vector<char> row_needs_artif(static_cast<std::size_t>(m_can), 0);
  std::vector<char> row_needs_surplus(static_cast<std::size_t>(m_can), 0);

  for (int i = 0; i < m_can; ++i) {
    const RowDesc& rd = rdesc[static_cast<std::size_t>(i)];
    double rhs = rd.rhs;
    if (rd.kind == RowDesc::Kind::EQ) {
      if (rhs < 0.0) {
        row_negate[static_cast<std::size_t>(i)] = 1;
        rhs = -rhs;
      }
      row_needs_artif[static_cast<std::size_t>(i)] = 1;
    } else if (rd.kind == RowDesc::Kind::LE) {
      if (rhs >= 0.0) {
        // a^T y + s = rhs, slack as feasible basis.
      } else {
        row_negate[static_cast<std::size_t>(i)] = 1;
        rhs = -rhs;
        row_needs_surplus[static_cast<std::size_t>(i)] = 1;
        row_needs_artif[static_cast<std::size_t>(i)] = 1;
      }
    } else {
      // GE: a^T y >= rhs.
      if (rhs <= 0.0) {
        row_negate[static_cast<std::size_t>(i)] = 1;
        rhs = -rhs;
        // After flip becomes <= |rhs|; slack feasible.
      } else {
        row_needs_surplus[static_cast<std::size_t>(i)] = 1;
        row_needs_artif[static_cast<std::size_t>(i)] = 1;
      }
    }
    b_can[static_cast<std::size_t>(i)] = rhs;
  }

  // Apply row sign flips to existing y triplets in-place.
  for (auto& t : trip) {
    int i, j; double v;
    std::tie(i, j, v) = t;
    if (row_negate[static_cast<std::size_t>(i)]) {
      std::get<2>(t) = -v;
    }
  }

  // Allocate aux columns. Layout: [y columns | slacks/surplus | artificials].
  // We add at most one slack OR one surplus per row, plus optional artificial.
  // "Aux non-basis": surplus column for ranged-style rows (negative coefficient, not basis).
  // "Aux basis":     either slack (non-artificial) or artificial.
  int n_aux_non_basis = 0;
  int n_aux_basis = 0;
  for (int i = 0; i < m_can; ++i) {
    if (row_needs_surplus[static_cast<std::size_t>(i)]) ++n_aux_non_basis;
    ++n_aux_basis;  // every row gets exactly one basis column
  }

  const int n_can = n_y + n_aux_non_basis + n_aux_basis;

  std::vector<int> basis_col_per_row(static_cast<std::size_t>(m_can), -1);
  std::vector<char> aux_is_artif(static_cast<std::size_t>(n_can), 0);
  std::vector<double> aux_cost(static_cast<std::size_t>(n_can), 0.0);

  // Copy y costs.
  for (int j = 0; j < n_y; ++j) {
    aux_cost[static_cast<std::size_t>(j)] = y_cost[static_cast<std::size_t>(j)];
  }

  // First emit non-basis aux (surplus columns for GE/flipped LE rows).
  int next_aux_non_basis = n_y;
  int next_aux_basis = n_y + n_aux_non_basis;
  for (int i = 0; i < m_can; ++i) {
    if (row_needs_surplus[static_cast<std::size_t>(i)]) {
      const int col = next_aux_non_basis++;
      trip.emplace_back(i, col, -1.0);
    }
  }
  // Then emit basis aux: artificial (if needed) else slack.
  for (int i = 0; i < m_can; ++i) {
    const int col = next_aux_basis++;
    trip.emplace_back(i, col, 1.0);
    basis_col_per_row[static_cast<std::size_t>(i)] = col;
    if (row_needs_artif[static_cast<std::size_t>(i)]) {
      aux_is_artif[static_cast<std::size_t>(col)] = 1;
    }
  }

  // Compute Big-M following the manual's heuristic: 100x to 1000x of max|c_j|.
  // Larger values cause catastrophic loss of significance during pivots; smaller
  // values fail to dominate genuine reduced costs. We use 1000 * max|c| with
  // floors and a small dependence on max|b| to keep artificials priced higher
  // than any plausible primal solution value.
  double cmax = 1.0;
  for (double v : y_cost) cmax = std::max(cmax, std::abs(v));
  double bmax = 1.0;
  for (double v : b_can) bmax = std::max(bmax, std::abs(v));
  cf.big_m = std::max(1.0e2, cmax * 1.0e3) * (1.0 + 0.1 * std::log10(1.0 + bmax));
  for (int j = 0; j < n_can; ++j) {
    if (aux_is_artif[static_cast<std::size_t>(j)]) {
      aux_cost[static_cast<std::size_t>(j)] = cf.big_m;
    }
  }

  // Build canonical SparseMatrix.
  cf.model.A = SparseMatrix::from_triplets(static_cast<std::size_t>(m_can),
                                           static_cast<std::size_t>(n_can), trip);
  cf.model.col_cost = std::move(aux_cost);
  cf.model.col_lower.assign(static_cast<std::size_t>(n_can), 0.0);
  cf.model.col_upper.assign(static_cast<std::size_t>(n_can), std::numeric_limits<double>::infinity());
  cf.model.row_lower = b_can;
  cf.model.row_upper = b_can;
  cf.model.sense = ObjSense::Minimize;
  cf.initial_basis = basis_col_per_row;
  cf.is_artificial = std::move(aux_is_artif);
  return cf;
}

static void load_mps_model(const std::string &path, LpModel &model) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open MPS file: " + path);
  }

  std::vector<std::string> row_order;
  std::vector<std::string> col_order;
  std::unordered_map<std::string, char> row_type;
  std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> col_coeff;
  std::unordered_map<std::string, double> rhs_values;
  std::unordered_map<std::string, double> row_ranges;
  std::unordered_map<std::string, double> lower_bounds;
  std::unordered_map<std::string, double> upper_bounds;
  std::string objective_name;

  parse_mps(in, row_order, col_order, row_type, col_coeff,
            rhs_values, row_ranges, lower_bounds, upper_bounds,
            objective_name);

  std::unordered_map<std::string, int> row_index;
  for (const auto &row_name : row_order) {
    auto it = row_type.find(row_name);
    if (it == row_type.end()) continue;
    if (it->second == 'N') continue;
    if (row_index.find(row_name) == row_index.end()) {
      row_index[row_name] = static_cast<int>(row_index.size());
    }
  }

  std::vector<std::tuple<int, int, double>> triplets;
  int num_cols = static_cast<int>(col_order.size());
  model.col_cost.assign(num_cols, 0.0);
  model.col_lower.assign(num_cols, 0.0);
  model.col_upper.assign(num_cols, std::numeric_limits<double>::infinity());

  for (int j = 0; j < num_cols; ++j) {
    const std::string &col_name = col_order[static_cast<std::size_t>(j)];
    auto it_coeff = col_coeff.find(col_name);
    if (it_coeff == col_coeff.end()) continue;
    for (const auto &entry : it_coeff->second) {
      const std::string &row_name = entry.first;
      double value = entry.second;
      auto it_row = row_index.find(row_name);
      if (it_row != row_index.end()) {
        triplets.emplace_back(it_row->second, j, value);
      } else {
        auto it_obj = row_type.find(row_name);
        if (it_obj != row_type.end() && it_obj->second == 'N') {
          model.col_cost[static_cast<std::size_t>(j)] = value;
        }
      }
    }
    auto it_lo = lower_bounds.find(col_name);
    if (it_lo != lower_bounds.end()) {
      model.col_lower[static_cast<std::size_t>(j)] = it_lo->second;
    }
    auto it_up = upper_bounds.find(col_name);
    if (it_up != upper_bounds.end()) {
      model.col_upper[static_cast<std::size_t>(j)] = it_up->second;
    }
  }

  int num_rows = static_cast<int>(row_index.size());
  model.row_lower.assign(num_rows, -std::numeric_limits<double>::infinity());
  model.row_upper.assign(num_rows, std::numeric_limits<double>::infinity());
  model.A = SparseMatrix::from_triplets(static_cast<std::size_t>(num_rows),
                                        static_cast<std::size_t>(num_cols), triplets);

  for (const auto &entry : row_index) {
    const std::string &row_name = entry.first;
    int row_idx = entry.second;
    double rhs_value = 0.0;
    auto it_rhs = rhs_values.find(row_name);
    if (it_rhs != rhs_values.end()) rhs_value = it_rhs->second;
    double range_value = std::numeric_limits<double>::quiet_NaN();
    auto it_range = row_ranges.find(row_name);
    if (it_range != row_ranges.end()) range_value = it_range->second;
    double lower = -std::numeric_limits<double>::infinity();
    double upper = std::numeric_limits<double>::infinity();
    auto it_type = row_type.find(row_name);
    if (it_type != row_type.end()) {
      effective_row_bounds(it_type->second, rhs_value, range_value, lower, upper);
    }
    model.row_lower[static_cast<std::size_t>(row_idx)] = lower;
    model.row_upper[static_cast<std::size_t>(row_idx)] = upper;
  }

  model.sense = ObjSense::Minimize;
}

// ---------- revised_dual_simplex.cpp ----------
RevisedDualSimplex::RevisedDualSimplex(LpModel model, LinearAlgebraBackendKind backend_kind,
                                           bool allow_phase_one)
    : model_(std::move(model)), backend_(make_backend(backend_kind)),
      allow_phase_one_(allow_phase_one) {
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

bool RevisedDualSimplex::set_initial_basis(std::vector<int> basis) {
  const int m = static_cast<int>(model_.A.rows());
  if (static_cast<int>(basis.size()) != m) return false;
  basis_cols_ = std::move(basis);
  return refactor();
}

bool RevisedDualSimplex::set_basis_scan_consecutive() {
  const int m = static_cast<int>(model_.A.rows());
  const int n = static_cast<int>(model_.A.cols());
  if (n < m) {
    return false;
  }
  basis_cols_.resize(static_cast<std::size_t>(m));
  for (int start = n - m; start >= 0; --start) {
    for (int k = 0; k < m; ++k) {
      basis_cols_[static_cast<std::size_t>(k)] = start + k;
    }
    if (refactor()) {
      return true;
    }
  }
  std::vector<int> pool(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    pool[static_cast<std::size_t>(j)] = j;
  }
  std::mt19937 rng(12345);
  const int max_trials = std::min(12000, std::max(4000, 15 * m * m));
  for (int trial = 0; trial < max_trials; ++trial) {
    std::shuffle(pool.begin(), pool.end(), rng);
    for (int k = 0; k < m; ++k) {
      basis_cols_[static_cast<std::size_t>(k)] = pool[static_cast<std::size_t>(k)];
    }
    if (refactor()) {
      return true;
    }
  }
  return false;
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

bool RevisedDualSimplex::phase_one(int max_iterations) {
  const int m = static_cast<int>(model_.A.rows());
  const int n = static_cast<int>(model_.A.cols());

  std::vector<double> b_phase(m);
  std::vector<char> row_flip(static_cast<std::size_t>(m), 0);
  for (int i = 0; i < m; ++i) {
    const double bval = model_.row_lower[static_cast<std::size_t>(i)];
    if (bval < 0.0) {
      row_flip[static_cast<std::size_t>(i)] = 1;
      b_phase[static_cast<std::size_t>(i)] = -bval;
    } else {
      b_phase[static_cast<std::size_t>(i)] = bval;
    }
  }

  std::vector<std::tuple<int, int, double>> triplets;
  triplets.reserve(model_.A.nnz() + static_cast<std::size_t>(m));
  for (int j = 0; j < n; ++j) {
    for (int p = model_.A.col_pointers()[static_cast<std::size_t>(j)];
         p < model_.A.col_pointers()[static_cast<std::size_t>(j + 1)]; ++p) {
      int i = model_.A.row_indices()[static_cast<std::size_t>(p)];
      double v = model_.A.values()[static_cast<std::size_t>(p)];
      if (row_flip[static_cast<std::size_t>(i)]) v = -v;
      triplets.emplace_back(i, j, v);
    }
  }
  for (int i = 0; i < m; ++i) {
    triplets.emplace_back(i, n + i, 1.0);
  }

  SparseMatrix A_phase = SparseMatrix::from_triplets(static_cast<std::size_t>(m),
                                                     static_cast<std::size_t>(n + m),
                                                     triplets);
  std::vector<double> c_phase(static_cast<std::size_t>(n + m), 0.0);
  double max_abs = 1.0;
  for (double v : model_.col_cost) {
    max_abs = std::max(max_abs, std::abs(v));
  }
  double M = max_abs * 1e8;
  for (int j = n; j < n + m; ++j) {
    c_phase[static_cast<std::size_t>(j)] = M;
  }

  LpModel phase_model;
  phase_model.A = std::move(A_phase);
  phase_model.col_cost = std::move(c_phase);
  phase_model.col_lower.assign(static_cast<std::size_t>(n + m), 0.0);
  phase_model.col_upper.assign(static_cast<std::size_t>(n + m), std::numeric_limits<double>::infinity());
  phase_model.row_lower = b_phase;
  phase_model.row_upper = b_phase;
  phase_model.sense = ObjSense::Minimize;

  RevisedDualSimplex phase_solver(std::move(phase_model), backend_->kind(), false);
  SolveStatus phase_status = phase_solver.solve(max_iterations);
  if (phase_status != SolveStatus::Optimal) {
    return false;
  }

  const std::vector<int>& phase_basis = phase_solver.basis_columns();
  const std::vector<double> phase_x_B = phase_solver.backend().solve(b_phase);
  std::vector<int> new_basis;
  new_basis.reserve(static_cast<std::size_t>(m));
  for (int i = 0; i < m; ++i) {
    int col = phase_basis[static_cast<std::size_t>(i)];
    if (col < n) {
      new_basis.push_back(col);
    } else {
      if (phase_x_B[static_cast<std::size_t>(i)] > 1e-9) {
        return false;
      }
    }
  }

  if (static_cast<int>(new_basis.size()) != m) {
    return false;
  }

  basis_cols_ = std::move(new_basis);
  return refactor();
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

double dot_sparse_column_with_vector(const SparseMatrix& A, int col, const std::vector<double>& v) {
  double s = 0.0;
  for (int p = A.col_pointers()[static_cast<std::size_t>(col)];
       p < A.col_pointers()[static_cast<std::size_t>(col + 1)]; ++p) {
    const int i = A.row_indices()[static_cast<std::size_t>(p)];
    s += A.values()[static_cast<std::size_t>(p)] * v[static_cast<std::size_t>(i)];
  }
  return s;
}

/// r = c_j - A_j^T pi (minimize). For maximize, dual sign flips: use A_j^T pi - c_j as "r" for pricing.
double reduced_cost_min_form(const LpModel& model, const std::vector<double>& pi, int j) {
  const double cj = model.col_cost[static_cast<std::size_t>(j)];
  const double zj = dot_sparse_column_with_vector(model.A, j, pi);
  if (model.sense == ObjSense::Maximize) {
    return zj - cj;
  }
  return cj - zj;
}

}  // namespace

// Two-phase primal revised simplex. Phase 1 minimizes the sum of artificial
// variables; Phase 2 then optimizes the original objective. This avoids the
// catastrophic ill-conditioning of the Big-M method while remaining faithful
// to the textbook revised simplex described in the project manual.
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
    const double rl = model_.row_lower[static_cast<std::size_t>(i)];
    const double ru = model_.row_upper[static_cast<std::size_t>(i)];
    if (!std::isfinite(rl) || !std::isfinite(ru)) {
      return SolveStatus::NotImplemented;
    }
    const double diff = std::abs(rl - ru);
    if (diff > 1e-9 * (1.0 + std::max(std::abs(rl), std::abs(ru)))) {
      return SolveStatus::NotImplemented;
    }
  }

  // ---- Initial basis ---------------------------------------------------------
  bool basis_ready = false;
  if (static_cast<int>(basis_cols_.size()) == m) {
    basis_ready = refactor();
  }
  if (!basis_ready) {
    if (!set_basis_scan_consecutive()) {
      try {
        set_basis_slack_tail();
      } catch (...) {
        return SolveStatus::FactorizationFailed;
      }
      if (!refactor()) {
        return SolveStatus::FactorizationFailed;
      }
    }
  }

  std::vector<double> b(model_.row_lower);

  // Make sure the artificial mask, if any, has the right dimension. If the
  // caller did not supply one we treat all columns as non-artificial.
  std::vector<char> artif_mask(static_cast<std::size_t>(n), 0);
  if (static_cast<int>(is_artificial_.size()) == n) {
    artif_mask = is_artificial_;
  }
  bool any_artif = false;
  for (char v : artif_mask) if (v) { any_artif = true; break; }

  // Stash original costs; Phase 1 uses an indicator cost on artificials, and
  // Phase 2 must drop the (big_M) artificial penalties or it degenerates back
  // to the numerically unstable Big-M method we are trying to avoid.
  const std::vector<double> stash_cost = model_.col_cost;
  std::vector<double> phase2_cost = stash_cost;
  for (std::size_t j = 0; j < phase2_cost.size(); ++j) {
    if (static_cast<int>(j) < n && j < artif_mask.size() && artif_mask[j]) {
      phase2_cost[j] = 0.0;
    }
  }

  // ---- Tolerances (Manual sec. 3.4) -----------------------------------------
  constexpr double kRcTol = 1e-7;       // dual feasibility tolerance epsilon_d
  constexpr double kPivotTol = 1e-7;    // numerical floor for a usable pivot
  constexpr double kHarrisEps = 1e-9;   // primal slack used by Harris pass 1
  constexpr int kBlandAfter = 60;       // anti-cycling kicks in after this many degenerate iters
  const int refactor_every = std::max(20, std::min(100, m));

  std::vector<double> x_B;
  std::vector<double> c_B(static_cast<std::size_t>(m));
  std::vector<char> is_basis(static_cast<std::size_t>(n), 0);
  std::vector<int> nonbasis;
  nonbasis.reserve(static_cast<std::size_t>(std::max(0, n - m)));

  // (Re)build derived basis bookkeeping. Bland's anti-cycling theorem requires
  // every nonbasic column to be a candidate, so we never exclude artificials
  // here -- their Phase 1 cost (1.0) is what stops them from re-entering once
  // primal feasibility is restored.
  auto rebuild_basis_view = [&]() {
    std::fill(is_basis.begin(), is_basis.end(), static_cast<char>(0));
    for (int i = 0; i < m; ++i) {
      const int bj = basis_cols_[static_cast<std::size_t>(i)];
      if (bj >= 0 && bj < n) {
        is_basis[static_cast<std::size_t>(bj)] = 1;
        c_B[static_cast<std::size_t>(i)] = model_.col_cost[static_cast<std::size_t>(bj)];
      } else {
        c_B[static_cast<std::size_t>(i)] = 0.0;
      }
    }
    nonbasis.clear();
    for (int j = 0; j < n; ++j) {
      if (!is_basis[static_cast<std::size_t>(j)]) nonbasis.push_back(j);
    }
  };

  auto safe_solve = [&](const std::vector<double>& rhs, std::vector<double>& out) -> bool {
    try { out = backend_->solve(rhs); }
    catch (...) {
      if (!refactor()) return false;
      try { out = backend_->solve(rhs); }
      catch (...) { return false; }
    }
    for (double v : out) {
      if (!std::isfinite(v)) {
        if (!refactor()) return false;
        try { out = backend_->solve(rhs); }
        catch (...) { return false; }
        break;
      }
    }
    return true;
  };

  auto safe_solve_transpose = [&](const std::vector<double>& rhs, std::vector<double>& out) -> bool {
    try { out = backend_->solve_transpose(rhs); }
    catch (...) {
      if (!refactor()) return false;
      try { out = backend_->solve_transpose(rhs); }
      catch (...) { return false; }
    }
    return true;
  };

  // Inner simplex loop. Returns one of {Optimal, Unbounded, FactorizationFailed,
  // NotImplemented}. NotImplemented is reserved for "iteration cap exhausted".
  // `forbid_artif_entry` disables artificial columns as entering candidates,
  // which is required in Phase 2 to avoid undoing primal feasibility.
  auto run_simplex = [&](int iter_budget, bool forbid_artif_entry) -> SolveStatus {
    int last_refactor = -refactor_every;  // force refactor on first iter
    int degenerate_streak = 0;
    int total_degenerate = 0;
    bool force_bland = false;
    // Cycle detector: ring buffer of recent (entering, leaving_col) pairs.
    constexpr int kCycleWindow = 32;
    std::vector<std::pair<int,int>> recent_pivots;
    recent_pivots.reserve(static_cast<std::size_t>(kCycleWindow));
    // Columns temporarily forbidden as entering candidates because previous
    // attempts produced a singular pivot. Cleared on every successful refactor
    // so genuinely useful columns get another chance after the basis changes.
    std::vector<char> forbid_entry(static_cast<std::size_t>(n), 0);
    // Wolfe-style perturbation. When the simplex stalls in degeneracy we
    // perturb b by tiny per-row offsets so all tie-breaking ratios become
    // strictly distinct. The perturbation is removed at exit so the reported
    // objective is computed from the original b. We use a deterministic
    // perturbation pattern (decreasing geometric series) so re-runs are
    // reproducible.
    const std::vector<double> b_orig = b;
    bool perturbed = false;
    int perturb_resets = 0;
    auto apply_perturbation = [&]() {
      if (perturbed) return;
      perturbed = true;
      double bscale = 0.0;
      for (double v : b_orig) bscale = std::max(bscale, std::abs(v));
      const double eps0 = std::max(1e-13, 1e-11 * (bscale + 1.0));
      // b'_i = b_i + eps0 * (1 + 1/2 + 1/4 + ... starting from 1/(2^i))
      for (int i = 0; i < m; ++i) {
        const double pi_i = eps0 / std::pow(2.0, i % 30);
        b[static_cast<std::size_t>(i)] = b_orig[static_cast<std::size_t>(i)] + pi_i;
      }
    };
    auto remove_perturbation = [&]() {
      if (!perturbed) return;
      perturbed = false;
      b = b_orig;
    };
    // RAII guard: ensure the perturbation is rolled back along every exit
    // path (return, exception) so x_B reflects the original right-hand side.
    struct ScopeGuard {
      std::function<void()> fn;
      ~ScopeGuard() { if (fn) fn(); }
    };
    ScopeGuard restore_b{[&]() { remove_perturbation(); }};

    for (int iter = 0; iter < iter_budget; ++iter) {
      if (iter - last_refactor >= refactor_every) {
        if (!refactor()) return SolveStatus::FactorizationFailed;
        last_refactor = iter;
      }

      if (!safe_solve(b, x_B)) return SolveStatus::FactorizationFailed;
      rebuild_basis_view();


      std::vector<double> pi;
      if (!safe_solve_transpose(c_B, pi)) return SolveStatus::FactorizationFailed;

      // CHUZC: pick entering column by Dantzig pricing, switching to Bland's
      // rule (smallest index) once we have spent too many iterations at zero
      // step length (degeneracy) -- this is the classical anti-cycling fallback.
      int entering = -1;
      double best_rc = -kRcTol;
      const bool use_bland = force_bland || (degenerate_streak >= kBlandAfter);
      // If Bland has been active for "long enough" without progress, perturb b
      // to break degeneracy (Wolfe's perturbation method). The threshold is
      // proportional to m so small problems perturb early and large problems
      // get more breathing room before paying the perturbation cost.
      if (use_bland && degenerate_streak == 4 * std::max(20, m) && !perturbed
          && perturb_resets < 5) {
        apply_perturbation();
        ++perturb_resets;
        // After perturbation, x_B must be recomputed; do so on the next
        // iteration's safe_solve. Reset Bland streak so we re-evaluate.
        degenerate_streak = 0;
        force_bland = false;
        recent_pivots.clear();
        continue;
      }
      if (use_bland) {
        for (int j : nonbasis) {
          if (forbid_artif_entry && artif_mask[static_cast<std::size_t>(j)]) continue;
          if (forbid_entry[static_cast<std::size_t>(j)]) continue;
          const double rc = reduced_cost_min_form(model_, pi, j);
          if (rc < -kRcTol) {
            entering = j;
            best_rc = rc;
            break;
          }
        }
      } else {
        for (int j : nonbasis) {
          if (forbid_artif_entry && artif_mask[static_cast<std::size_t>(j)]) continue;
          if (forbid_entry[static_cast<std::size_t>(j)]) continue;
          const double rc = reduced_cost_min_form(model_, pi, j);
          if (rc < best_rc) {
            best_rc = rc;
            entering = j;
          }
        }
      }
      if (entering < 0) {
        // If we were running with a perturbed b, the current basis is optimal
        // for the perturbed LP but may not be for the true LP. Drop the
        // perturbation, refactor, and let the simplex continue. This usually
        // costs only a handful of pivots since we are already near optimum.
        if (perturbed) {
          remove_perturbation();
          if (!refactor()) return SolveStatus::FactorizationFailed;
          last_refactor = iter;
          degenerate_streak = 0;
          force_bland = false;
          recent_pivots.clear();
          continue;
        }
        return SolveStatus::Optimal;
      }

      std::vector<double> a_col = build_dense_column_from_sparse(model_.A, entering);
      std::vector<double> d;
      if (!safe_solve(a_col, d)) return SolveStatus::FactorizationFailed;

      // Ratio test. In Bland mode we use the strict minimum-ratio rule so the
      // anti-cycling theorem applies; otherwise we use Harris' two-pass test
      // (Manual sec. 3.4) to pick the largest pivot among quasi-tied rows.
      int leaving_basis_index = -1;
      double leaving_pivot = 0.0;
      int leaving_col_index = std::numeric_limits<int>::max();
      const double pivot_floor_strict = kPivotTol;

      if (use_bland) {
        // Strict Bland's rule for the leaving variable: among rows with d_i > 0
        // and the minimum ratio, pick the smallest basis-column index. We use
        // a moderate pivot floor (>= kPivotTol) so noise-magnitude positives
        // cannot win the tie-break -- this would defeat the anti-cycling
        // theorem in floating-point arithmetic.
        const double pivot_floor = std::max(1e-9, kPivotTol);
        double theta_strict = std::numeric_limits<double>::infinity();
        for (int i = 0; i < m; ++i) {
          const double di = d[static_cast<std::size_t>(i)];
          if (di <= pivot_floor) continue;
          const double xi = std::max(0.0, x_B[static_cast<std::size_t>(i)]);
          const double r = xi / di;
          if (r < theta_strict) theta_strict = r;
        }
        if (!std::isfinite(theta_strict)) return SolveStatus::Unbounded;
        const double tie_tol = std::max(1e-12, 1e-9 * std::abs(theta_strict));
        for (int i = 0; i < m; ++i) {
          const double di = d[static_cast<std::size_t>(i)];
          if (di <= pivot_floor) continue;
          const double xi = std::max(0.0, x_B[static_cast<std::size_t>(i)]);
          const double r = xi / di;
          if (r > theta_strict + tie_tol) continue;
          const int col = basis_cols_[static_cast<std::size_t>(i)];
          if (leaving_basis_index < 0 || col < leaving_col_index) {
            leaving_basis_index = i;
            leaving_pivot = di;
            leaving_col_index = col;
          }
        }
      } else {
        // Harris pass 1: theta_max with tiny primal slack to absorb noise.
        double theta_max = std::numeric_limits<double>::infinity();
        double pivot_floor = pivot_floor_strict;
        for (int pass = 0; pass < 2 && !std::isfinite(theta_max); ++pass) {
          if (pass == 1) pivot_floor = 1e-12;
          for (int i = 0; i < m; ++i) {
            const double di = d[static_cast<std::size_t>(i)];
            if (di <= pivot_floor) continue;
            const double xi = x_B[static_cast<std::size_t>(i)];
            const double ratio = (std::max(0.0, xi) + kHarrisEps) / di;
            if (ratio < theta_max) theta_max = ratio;
          }
        }
        if (!std::isfinite(theta_max)) return SolveStatus::Unbounded;

        // Pass 2: pick the largest pivot among admissible rows; prefer
        // artificials leaving on near-ties.
        for (int i = 0; i < m; ++i) {
          const double di = d[static_cast<std::size_t>(i)];
          if (di <= pivot_floor) continue;
          const double xi = x_B[static_cast<std::size_t>(i)];
          const double strict_ratio = std::max(0.0, xi) / di;
          if (strict_ratio > theta_max + 1e-12) continue;
          const int col = basis_cols_[static_cast<std::size_t>(i)];
          const bool i_is_a =
              (col >= 0 && col < n) && artif_mask[static_cast<std::size_t>(col)];
          const bool prev_is_a =
              (leaving_col_index >= 0 && leaving_col_index < n)
              && artif_mask[static_cast<std::size_t>(leaving_col_index)];
          bool take = (leaving_basis_index < 0);
          if (!take) {
            if (di > leaving_pivot * 1.05) {
              take = true;
            } else if (di > leaving_pivot * 0.95) {
              if (i_is_a && !prev_is_a) take = true;
              else if (i_is_a == prev_is_a && col < leaving_col_index) take = true;
            }
          }
          if (take) {
            leaving_basis_index = i;
            leaving_pivot = di;
            leaving_col_index = col;
          }
        }
      }
      if (leaving_basis_index < 0) return SolveStatus::Unbounded;

      const double xi_leave = x_B[static_cast<std::size_t>(leaving_basis_index)];
      const double theta_actual = std::max(0.0, xi_leave) / leaving_pivot;
      if (theta_actual <= 1e-12) {
        ++degenerate_streak;
        ++total_degenerate;
      } else {
        degenerate_streak = 0;
      }

      // Cycle detection: if the same (entering, leaving_col) pair shows up too
      // often in the recent window, we are in a true cycle (which mixes
      // degenerate and non-degenerate pivots and so the simple degenerate
      // streak heuristic misses it). Switch to Bland's rule permanently.
      if (!force_bland) {
        const std::pair<int,int> key{entering, leaving_col_index};
        int hits = 0;
        for (const auto& p : recent_pivots) if (p == key) ++hits;
        if (hits >= 3) {
          force_bland = true;
        }
        if (static_cast<int>(recent_pivots.size()) >= kCycleWindow) {
          recent_pivots.erase(recent_pivots.begin());
        }
        recent_pivots.push_back(key);
      }

      const int prev_col = basis_cols_[static_cast<std::size_t>(leaving_basis_index)];
      basis_cols_[static_cast<std::size_t>(leaving_basis_index)] = entering;
      if (!refactor()) {
        // Numerically singular pivot. Roll back to the previous basis (which
        // was non-singular), then forbid this entering column for the rest of
        // the run -- attempting the same pivot again will only fail and the
        // "any-row-with-positive-d" recovery used previously corrupts x_B with
        // arbitrary sign, blowing up the solve. The forbidden-column list is
        // erased after iter_budget; in practice only a handful of columns get
        // marked because each marking changes pi on the next iteration.
        basis_cols_[static_cast<std::size_t>(leaving_basis_index)] = prev_col;
        if (!refactor()) return SolveStatus::FactorizationFailed;
        forbid_entry[static_cast<std::size_t>(entering)] = 1;
        // Don't count this as a real iteration toward degeneracy/cycle stats.
        --iter;
        continue;
      }
      // Successful pivot -- clear all forbidden columns; the basis has changed
      // and previously-bad pivots may now be safe.
      std::fill(forbid_entry.begin(), forbid_entry.end(), 0);
      last_refactor = iter;
    }
    return SolveStatus::NotImplemented;  // iteration cap reached
  };

  // ---- Phase 1: drive artificials to zero -----------------------------------
  if (any_artif) {
    std::vector<double> phase1_cost(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
      if (artif_mask[static_cast<std::size_t>(j)]) phase1_cost[static_cast<std::size_t>(j)] = 1.0;
    }
    model_.col_cost = phase1_cost;

    // Phase 1 frequently needs more iterations than Phase 2 (it has to drive
    // every artificial to zero through highly degenerate pivots), so give it
    // 3/4 of the budget rather than splitting evenly.
    const int p1_budget = std::max(5000, (max_iterations * 3) / 4);
    SolveStatus p1 = run_simplex(p1_budget, /*forbid_artif_entry=*/false);
    if (p1 == SolveStatus::FactorizationFailed) {
      model_.col_cost = stash_cost;
      return SolveStatus::FactorizationFailed;
    }
    // Compute Phase 1 objective: sum of basic artificials (which equals their
    // primal values since their cost is 1).
    if (!safe_solve(b, x_B)) {
      model_.col_cost = stash_cost;
      return SolveStatus::FactorizationFailed;
    }
    double phase1_obj = 0.0;
    for (int i = 0; i < m; ++i) {
      const int col = basis_cols_[static_cast<std::size_t>(i)];
      if (col >= 0 && col < n && artif_mask[static_cast<std::size_t>(col)]) {
        phase1_obj += std::max(0.0, x_B[static_cast<std::size_t>(i)]);
      }
    }
    if (phase1_obj > 1e-6 * (1.0 + std::abs(phase1_obj))) {
      model_.col_cost = stash_cost;
      return SolveStatus::Infeasible;
    }

    // Try to drive any remaining (zero-valued) artificials out of the basis
    // by pivoting them against any non-artificial column with non-zero
    // d-coefficient. This is purely cosmetic for Phase 2 stability.
    for (int i = 0; i < m; ++i) {
      const int col = basis_cols_[static_cast<std::size_t>(i)];
      if (col < 0 || col >= n || !artif_mask[static_cast<std::size_t>(col)]) continue;
      // Find any non-artificial nonbasic column whose row i entry of B^{-1} A
      // is non-zero. Cheap probe: try each candidate via FTRAN.
      bool swapped = false;
      for (int j = 0; j < n; ++j) {
        if (is_basis[static_cast<std::size_t>(j)]) continue;
        if (artif_mask[static_cast<std::size_t>(j)]) continue;
        std::vector<double> aj = build_dense_column_from_sparse(model_.A, j);
        std::vector<double> dj;
        if (!safe_solve(aj, dj)) break;
        if (std::abs(dj[static_cast<std::size_t>(i)]) > kPivotTol) {
          basis_cols_[static_cast<std::size_t>(i)] = j;
          if (refactor()) { swapped = true; break; }
          basis_cols_[static_cast<std::size_t>(i)] = col;
        }
      }
      (void)swapped;
    }

    model_.col_cost = phase2_cost;
    if (!refactor()) return SolveStatus::FactorizationFailed;
  } else {
    model_.col_cost = phase2_cost;
  }

  // ---- Phase 2: optimize the original objective (artificial costs zeroed) ----
  SolveStatus p2 = run_simplex(max_iterations, /*forbid_artif_entry=*/true);
  if (p2 == SolveStatus::Optimal) return SolveStatus::Optimal;
  if (p2 == SolveStatus::Unbounded) return SolveStatus::Unbounded;
  if (p2 == SolveStatus::FactorizationFailed) return SolveStatus::FactorizationFailed;

  // Iteration cap reached -- accept if reduced costs are already feasible.
  try {
    if (safe_solve(b, x_B)) {
      rebuild_basis_view();
      std::vector<double> pi_final;
      if (safe_solve_transpose(c_B, pi_final)) {
        double worst_rc = 0.0;
        for (int j : nonbasis) {
          worst_rc = std::min(worst_rc, reduced_cost_min_form(model_, pi_final, j));
        }
        if (worst_rc >= -1e-6) return SolveStatus::Optimal;
      }
    }
  } catch (...) {}
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
      double rl = row_lower[static_cast<std::size_t>(i)];
      double ru = row_upper[static_cast<std::size_t>(i)];
      if (rl > 0.0 || ru < 0.0) {
        out.infeasible = true;
        out.message = "presolve: infeasible empty row";
        return out;
      }
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
      if (c[static_cast<std::size_t>(j)] < 0.0) {
        out.unbounded = true;
        out.message = "presolve: unbounded empty column";
        return out;
      }
      out.fixed_value_orig[static_cast<std::size_t>(j)] = 0.0;
      out.is_fixed_orig[static_cast<std::size_t>(j)] = 1;
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
        if (xv < -kTol) {
          out.infeasible = true;
          out.message = "presolve: infeasible singleton row";
          return out;
        }
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



int main(int argc, char* argv[]) {
  using namespace sparse_lp;

  std::vector<std::string> mps_files;
  if (argc == 2) {
    mps_files.push_back(argv[1]);
  } else {
    // Solve all .mps files in netlib directory
    for (const auto& entry : std::filesystem::directory_iterator("netlib")) {
      if (entry.path().extension() == ".mps") {
        mps_files.push_back(entry.path().string());
      }
    }
  }

  int total = 0, optimal = 0, infeas = 0, unbnd = 0, other = 0;

  for (const auto& mps_file : mps_files) {
    std::cout << "\n========================================\n";
    std::cout << "Solving " << mps_file << "\n";
    ++total;

    LpModel model;
    try {
      load_mps_model(mps_file, model);
      const int m0 = static_cast<int>(model.A.rows());
      const int n0 = static_cast<int>(model.A.cols());
      const std::size_t nnz0 = model.A.nnz();
      std::cout << "Loaded: " << m0 << "x" << n0 << " matrix, nnz=" << nnz0 << "\n";

      if (nnz0 > 5000000 || n0 > 50000 || m0 > 50000) {
        std::cout << "Status: SKIPPED (too large)\n";
        ++other;
        continue;
      }
    } catch (const std::exception &ex) {
      std::cerr << "ERROR: Failed to load MPS: " << ex.what() << "\n";
      ++other;
      continue;
    } catch (...) {
      std::cerr << "ERROR: Unknown exception during MPS load\n";
      ++other;
      continue;
    }

    try {
      const ObjSense sense_orig = model.sense;
      // Convert maximize to minimize by negating costs (we solve in min form internally).
      if (sense_orig == ObjSense::Maximize) {
        for (auto& v : model.col_cost) v = -v;
        model.sense = ObjSense::Minimize;
      }

      CanonicalForm cf = to_canonical_form(model);
      const int m_can = static_cast<int>(cf.model.A.rows());
      const int n_can = static_cast<int>(cf.model.A.cols());
      std::cout << "Canonical: " << m_can << "x" << n_can
                << " matrix, nnz=" << cf.model.A.nnz()
                << ", big_M=" << cf.big_m << "\n";

      RevisedDualSimplex solver(std::move(cf.model), LinearAlgebraBackendKind::Native,
                                /*allow_phase_one=*/false);
      solver.set_is_artificial(cf.is_artificial);
      const bool ok_init = solver.set_initial_basis(cf.initial_basis);
      if (!ok_init) {
        std::cout << "Status: FACTORIZATION_FAILED (initial basis)\n";
        ++other;
        continue;
      }
      const int max_iter = std::max(20000, 50 * (m_can + n_can));
      SolveStatus status = solver.solve(max_iter);

      // Detect infeasibility via residual artificial variables.
      bool any_artif_in_basis_positive = false;
      if (status == SolveStatus::Optimal) {
        try {
          const std::vector<int>& bcols = solver.basis_columns();
          const std::vector<double> x_B = solver.backend().solve(solver.model().row_lower);
          for (int i = 0; i < m_can; ++i) {
            const int col = bcols[static_cast<std::size_t>(i)];
            if (col >= 0 && col < n_can &&
                cf.is_artificial[static_cast<std::size_t>(col)] &&
                x_B[static_cast<std::size_t>(i)] > 1e-6) {
              any_artif_in_basis_positive = true;
              break;
            }
          }
        } catch (...) {}
      }

      std::cout << "Status: ";
      if (status == SolveStatus::Optimal) {
        if (any_artif_in_basis_positive) {
          std::cout << "INFEASIBLE";
          ++infeas;
        } else {
          double obj_val = cf.obj_offset;
          try {
            const std::vector<int>& bcols = solver.basis_columns();
            const std::vector<double> x_B = solver.backend().solve(solver.model().row_lower);
            for (int i = 0; i < m_can; ++i) {
              const int col = bcols[static_cast<std::size_t>(i)];
              if (col >= 0 && col < n_can) {
                obj_val += solver.model().col_cost[static_cast<std::size_t>(col)]
                          * x_B[static_cast<std::size_t>(i)];
              }
            }
            if (sense_orig == ObjSense::Maximize) {
              obj_val = -obj_val;
            }
          } catch (...) {}
          std::cout << "OPTIMAL  obj=" << obj_val;
          ++optimal;
        }
      } else if (status == SolveStatus::Unbounded) {
        std::cout << "UNBOUNDED";
        ++unbnd;
      } else if (status == SolveStatus::FactorizationFailed) {
        std::cout << "FACTORIZATION_FAILED";
        ++other;
      } else if (status == SolveStatus::NotImplemented) {
        std::cout << "NOT_CONVERGED";
        ++other;
      } else if (status == SolveStatus::DimensionError) {
        std::cout << "DIMENSION_ERROR";
        ++other;
      } else if (status == SolveStatus::Infeasible) {
        std::cout << "INFEASIBLE";
        ++infeas;
      } else {
        std::cout << "UNKNOWN";
        ++other;
      }
      std::cout << "\n";

    } catch (const std::exception &ex) {
      std::cerr << "ERROR: Solver failed: " << ex.what() << "\n";
      ++other;
    } catch (...) {
      std::cerr << "ERROR: Unknown exception\n";
      ++other;
    }
  }

  std::cout << "\n========================================\n";
  std::cout << "Summary: " << optimal << " OPTIMAL, " << infeas << " INFEASIBLE, "
            << unbnd << " UNBOUNDED, " << other << " other  (total " << total << ")\n";
  return 0;
}
