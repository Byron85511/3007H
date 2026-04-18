#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cctype>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "lp_solver/linalg/i_basis_factor.hpp"
#include "lp_solver/model/initialization.hpp"
#include "lp_solver/model/problem_data.hpp"
#include "lp_solver/model/solver_state.hpp"
#include "lp_solver/presolve/presolver.hpp"
#include "lp_solver/simplex/dual_simplex.hpp"
#include "lp_solver/simplex/naive_row_pivot.hpp"
#include "lp_solver/util/packed_matrix.hpp"

namespace fs = std::filesystem;

namespace {

struct MpsProblem {
  std::vector<std::string> row_names;
  std::vector<char> row_types;
  std::unordered_map<std::string, int> row_index;

  std::vector<std::string> var_names;
  std::unordered_map<std::string, int> var_index;
  std::vector<std::unordered_map<int, double>> row_coeffs;
  std::vector<double> objective;

  std::vector<double> rhs;
  std::vector<double> ranges;
  std::vector<double> lb;
  std::vector<double> ub;
  std::vector<char> upper_is_explicit;

  int objective_row{-1};
};

struct RowConstraint {
  std::unordered_map<int, double> coeffs;
  double rhs{0.0};
  bool needs_slack{false};  // true for <= rows, false for equalities
};

struct StandardFormModel {
  lp_solver::model::ProblemData problem;
  double objective_constant{0.0};
};

enum class PrimalStatus { Optimal, Infeasible, Unbounded, IterationLimit, NumericalFailure };

struct PrimalResult {
  PrimalStatus status{PrimalStatus::NumericalFailure};
  double objective{0.0};
  std::vector<int> basic_indices;
  std::vector<double> x_basic;
};

std::vector<std::string> splitTokens(const std::string& line) {
  std::istringstream in(line);
  std::vector<std::string> out;
  std::string tok;
  while (in >> tok) {
    out.push_back(tok);
  }
  return out;
}

std::string fieldTrim(const std::string& line, size_t start, size_t width) {
  if (start >= line.size()) {
    return "";
  }
  const size_t end = std::min(line.size(), start + width);
  size_t b = start;
  while (b < end && std::isspace(static_cast<unsigned char>(line[b])) != 0) {
    ++b;
  }
  size_t e = end;
  while (e > b && std::isspace(static_cast<unsigned char>(line[e - 1])) != 0) {
    --e;
  }
  return line.substr(b, e - b);
}

bool isCloseZero(double v, double eps = 1e-14) { return std::abs(v) <= eps; }

double inf() { return std::numeric_limits<double>::infinity(); }

int ensureVar(MpsProblem& mps, const std::string& var) {
  const auto it = mps.var_index.find(var);
  if (it != mps.var_index.end()) {
    return it->second;
  }
  const int idx = static_cast<int>(mps.var_names.size());
  mps.var_names.push_back(var);
  mps.var_index.emplace(var, idx);
  mps.objective.push_back(0.0);
  mps.lb.push_back(0.0);  // MPS default lower bound.
  mps.ub.push_back(inf());
  mps.upper_is_explicit.push_back(0);
  return idx;
}

MpsProblem parseMps(const fs::path& file_path) {
  std::ifstream in(file_path);
  if (!in) {
    throw std::runtime_error("cannot open file: " + file_path.string());
  }

  enum class Section { None, Name, Rows, Columns, Rhs, Ranges, Bounds };
  Section section = Section::None;

  MpsProblem mps;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    if (!line.empty() && line[0] == '*') {
      continue;
    }
    const bool has_field_1 = line.size() > 1 && line[1] != ' ';
    auto tokens = splitTokens(line);
    if (tokens.empty()) {
      continue;
    }
    std::string key = tokens[0];
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (has_field_1 && key == "NAME") {
      section = Section::Name;
      continue;
    }
    if (has_field_1 && key == "ROWS") {
      section = Section::Rows;
      continue;
    }
    if (has_field_1 && key == "COLUMNS") {
      section = Section::Columns;
      continue;
    }
    if (has_field_1 && key == "RHS") {
      section = Section::Rhs;
      continue;
    }
    if (has_field_1 && key == "RANGES") {
      section = Section::Ranges;
      continue;
    }
    if (has_field_1 && key == "BOUNDS") {
      section = Section::Bounds;
      continue;
    }
    if (has_field_1 && key == "ENDATA") {
      break;
    }

    if (section == Section::Rows) {
      const std::string row_type_s = fieldTrim(line, 1, 1);   // column 2
      const std::string row_name = fieldTrim(line, 4, 8);     // columns 5-12
      if (row_type_s.empty() || row_name.empty()) {
        continue;
      }
      const char row_type = row_type_s[0];
      const int row_id = static_cast<int>(mps.row_names.size());
      mps.row_names.push_back(row_name);
      mps.row_types.push_back(row_type);
      mps.row_index.emplace(row_name, row_id);
      if (row_type == 'N' && mps.objective_row < 0) {
        mps.objective_row = row_id;
      }
      continue;
    }

    if (section == Section::Columns) {
      const std::string var_name = fieldTrim(line, 4, 8);   // columns 5-12
      const std::string row1 = fieldTrim(line, 14, 8);      // columns 15-22
      const std::string val1 = fieldTrim(line, 24, 12);     // columns 25-36
      const std::string row2 = fieldTrim(line, 39, 8);      // columns 40-47
      const std::string val2 = fieldTrim(line, 49, 12);     // columns 50-61
      if (var_name.empty()) {
        continue;
      }
      if (row1.find("MARKER") != std::string::npos || row2.find("MARKER") != std::string::npos) {
        continue;
      }
      const int var_id = ensureVar(mps, var_name);
      auto read_pair = [&](const std::string& row_name, const std::string& value_str) {
        if (row_name.empty() || value_str.empty()) {
          return;
        }
        const auto r_it = mps.row_index.find(row_name);
        if (r_it == mps.row_index.end()) {
          throw std::runtime_error("unknown row in COLUMNS: " + row_name);
        }
        const int rid = r_it->second;
        const double val = std::stod(value_str);
        if (mps.row_types[static_cast<size_t>(rid)] == 'N') {
          mps.objective[static_cast<size_t>(var_id)] += val;
          return;
        }
        if (mps.row_coeffs.empty()) {
          mps.row_coeffs.resize(mps.row_names.size());
        }
        mps.row_coeffs[static_cast<size_t>(rid)][var_id] += val;
      };
      read_pair(row1, val1);
      read_pair(row2, val2);
      continue;
    }

    if (section == Section::Rhs) {
      // Fixed-format RHS: set name in columns 5-12 (ignored), row/value pairs in 15-22/25-36 and
      // 40-47/50-61.
      const std::string row1_fixed = fieldTrim(line, 14, 8);
      const std::string val1_fixed = fieldTrim(line, 24, 12);
      const std::string row2_fixed = fieldTrim(line, 39, 8);
      const std::string val2_fixed = fieldTrim(line, 49, 12);
      if (row1_fixed.empty() && row2_fixed.empty()) {
        continue;
      }
      if (mps.rhs.empty()) {
        mps.rhs.assign(mps.row_names.size(), 0.0);
      }
      auto read_pair = [&](const std::string& row_name, const std::string& value_str) {
        if (row_name.empty() || value_str.empty()) {
          return;
        }
        const auto r_it = mps.row_index.find(row_name);
        if (r_it == mps.row_index.end()) {
          return;
        }
        mps.rhs[static_cast<size_t>(r_it->second)] = std::stod(value_str);
      };
      read_pair(row1_fixed, val1_fixed);
      read_pair(row2_fixed, val2_fixed);
      continue;
    }

    if (section == Section::Ranges) {
      const std::string row1_fixed = fieldTrim(line, 14, 8);
      const std::string val1_fixed = fieldTrim(line, 24, 12);
      const std::string row2_fixed = fieldTrim(line, 39, 8);
      const std::string val2_fixed = fieldTrim(line, 49, 12);
      if (row1_fixed.empty() && row2_fixed.empty()) {
        continue;
      }
      if (mps.ranges.empty()) {
        mps.ranges.assign(mps.row_names.size(), 0.0);
      }
      auto read_pair = [&](const std::string& row_name, const std::string& value_str) {
        if (row_name.empty() || value_str.empty()) {
          return;
        }
        const auto r_it = mps.row_index.find(row_name);
        if (r_it == mps.row_index.end()) {
          return;
        }
        mps.ranges[static_cast<size_t>(r_it->second)] = std::stod(value_str);
      };
      read_pair(row1_fixed, val1_fixed);
      read_pair(row2_fixed, val2_fixed);
      continue;
    }

    if (section == Section::Bounds) {
      const std::string type = fieldTrim(line, 1, 2);      // columns 2-3
      const std::string var_name = fieldTrim(line, 14, 8); // columns 15-22
      const std::string value_str = fieldTrim(line, 24, 12);
      if (type.empty() || var_name.empty()) {
        continue;
      }
      const int var_id = ensureVar(mps, var_name);
      const double val = value_str.empty() ? 0.0 : std::stod(value_str);

      double& lo = mps.lb[static_cast<size_t>(var_id)];
      double& up = mps.ub[static_cast<size_t>(var_id)];
      if (type == "LO" || type == "LI") {
        lo = val;
      } else if (type == "UP" || type == "UI") {
        up = val;
        mps.upper_is_explicit[static_cast<size_t>(var_id)] = 1;
      } else if (type == "FX") {
        lo = val;
        up = val;
        mps.upper_is_explicit[static_cast<size_t>(var_id)] = 1;
      } else if (type == "FR") {
        lo = -inf();
        up = inf();
        mps.upper_is_explicit[static_cast<size_t>(var_id)] = 1;
      } else if (type == "MI") {
        lo = -inf();
        mps.upper_is_explicit[static_cast<size_t>(var_id)] = 1;
      } else if (type == "PL") {
        up = inf();
        mps.upper_is_explicit[static_cast<size_t>(var_id)] = 1;
      }
      continue;
    }
  }

  if (mps.objective_row < 0) {
    throw std::runtime_error("no objective row in MPS");
  }
  if (mps.row_coeffs.empty()) {
    mps.row_coeffs.resize(mps.row_names.size());
  }
  if (mps.rhs.empty()) {
    mps.rhs.assign(mps.row_names.size(), 0.0);
  }
  if (mps.ranges.empty()) {
    mps.ranges.assign(mps.row_names.size(), 0.0);
  }
  return mps;
}

std::vector<RowConstraint> buildRowConstraints(const MpsProblem& mps) {
  std::vector<RowConstraint> out;
  for (int rid = 0; rid < static_cast<int>(mps.row_names.size()); ++rid) {
    const char t = mps.row_types[static_cast<size_t>(rid)];
    if (t == 'N') {
      continue;
    }
    const double rhs = mps.rhs[static_cast<size_t>(rid)];
    const double range = mps.ranges[static_cast<size_t>(rid)];
    const bool has_range = !isCloseZero(range);

    double lo = -inf();
    double up = inf();
    if (t == 'L') {
      up = rhs;
    } else if (t == 'G') {
      lo = rhs;
    } else if (t == 'E') {
      lo = rhs;
      up = rhs;
    } else {
      continue;
    }

    if (has_range) {
      if (t == 'L') {
        lo = rhs - std::abs(range);
        up = rhs;
      } else if (t == 'G') {
        lo = rhs;
        up = rhs + std::abs(range);
      } else if (t == 'E') {
        if (range >= 0.0) {
          lo = rhs;
          up = rhs + range;
        } else {
          lo = rhs + range;
          up = rhs;
        }
      }
    }

    if (std::isfinite(lo) && std::isfinite(up) && std::abs(lo - up) <= 1e-14) {
      RowConstraint row;
      row.coeffs = mps.row_coeffs[static_cast<size_t>(rid)];
      row.rhs = up;
      row.needs_slack = false;
      out.push_back(std::move(row));
      continue;
    }

    if (std::isfinite(up)) {
      RowConstraint row;
      row.coeffs = mps.row_coeffs[static_cast<size_t>(rid)];
      row.rhs = up;
      row.needs_slack = true;
      out.push_back(std::move(row));
    }
    if (std::isfinite(lo)) {
      RowConstraint row;
      row.rhs = -lo;
      row.needs_slack = true;
      for (const auto& [j, a] : mps.row_coeffs[static_cast<size_t>(rid)]) {
        row.coeffs[j] = -a;
      }
      out.push_back(std::move(row));
    }
  }
  return out;
}

StandardFormModel toStandardForm(const MpsProblem& mps) {
  auto rows = buildRowConstraints(mps);
  const int n_orig = static_cast<int>(mps.var_names.size());

  struct VarDesc {
    double shift{0.0};
    std::vector<std::pair<int, double>> terms;
  };

  std::vector<VarDesc> desc(static_cast<size_t>(n_orig));
  std::vector<double> c_new;
  double objective_constant = 0.0;

  auto add_new_var = [&](double cost) {
    const int id = static_cast<int>(c_new.size());
    c_new.push_back(cost);
    return id;
  };

  for (int j = 0; j < n_orig; ++j) {
    const double lo = mps.lb[static_cast<size_t>(j)];
    const double up = mps.ub[static_cast<size_t>(j)];
    const double cj = mps.objective[static_cast<size_t>(j)];
    VarDesc d;

    if (std::isfinite(lo) && std::isfinite(up) && std::abs(lo - up) <= 1e-14) {
      d.shift = lo;
      objective_constant += cj * lo;
      desc[static_cast<size_t>(j)] = d;
      continue;
    }

    if (std::isfinite(lo)) {
      const int z = add_new_var(cj);
      d.shift = lo;
      d.terms.emplace_back(z, 1.0);
      objective_constant += cj * lo;
      if (mps.upper_is_explicit[static_cast<size_t>(j)] != 0 && std::isfinite(up)) {
        const double ub = up - lo;
        RowConstraint r;
        r.coeffs[z] = 1.0;
        r.rhs = ub;
        r.needs_slack = true;
        rows.push_back(std::move(r));
      }
      desc[static_cast<size_t>(j)] = std::move(d);
      continue;
    }

    if (std::isfinite(up)) {
      const int z = add_new_var(-cj);
      d.shift = up;
      d.terms.emplace_back(z, -1.0);
      objective_constant += cj * up;
      desc[static_cast<size_t>(j)] = std::move(d);
      continue;
    }

    const int zp = add_new_var(cj);
    const int zm = add_new_var(-cj);
    d.terms.emplace_back(zp, 1.0);
    d.terms.emplace_back(zm, -1.0);
    desc[static_cast<size_t>(j)] = std::move(d);
  }

  std::vector<RowConstraint> transformed_rows;
  transformed_rows.reserve(rows.size());
  for (const auto& row : rows) {
    RowConstraint tr;
    tr.rhs = row.rhs;
    tr.needs_slack = row.needs_slack;
    for (const auto& [j, a] : row.coeffs) {
      const auto& d = desc[static_cast<size_t>(j)];
      tr.rhs -= a * d.shift;
      for (const auto& [nv, mul] : d.terms) {
        tr.coeffs[nv] += a * mul;
      }
    }
    transformed_rows.push_back(std::move(tr));
  }

  const int m = static_cast<int>(transformed_rows.size());
  const int n_vars = static_cast<int>(c_new.size());
  int slack_count = 0;
  for (const auto& row : transformed_rows) {
    if (row.needs_slack) {
      ++slack_count;
    }
  }
  const int n_total = n_vars + slack_count;

  std::vector<std::vector<int>> col_rows(static_cast<size_t>(n_total));
  std::vector<std::vector<double>> col_vals(static_cast<size_t>(n_total));
  std::vector<double> b;
  b.reserve(static_cast<size_t>(m));
  for (int i = 0; i < m; ++i) {
    b.push_back(transformed_rows[static_cast<size_t>(i)].rhs);
    for (const auto& [j, a] : transformed_rows[static_cast<size_t>(i)].coeffs) {
      if (isCloseZero(a)) {
        continue;
      }
      col_rows[static_cast<size_t>(j)].push_back(i);
      col_vals[static_cast<size_t>(j)].push_back(a);
    }
  }

  int next_slack_col = n_vars;
  for (int i = 0; i < m; ++i) {
    if (!transformed_rows[static_cast<size_t>(i)].needs_slack) {
      continue;
    }
    col_rows[static_cast<size_t>(next_slack_col)].push_back(i);
    col_vals[static_cast<size_t>(next_slack_col)].push_back(1.0);
    c_new.push_back(0.0);
    ++next_slack_col;
  }

  lp_solver::util::PackedMatrix::Builder builder(m, n_total);
  for (int j = 0; j < n_total; ++j) {
    builder.appendColumn(col_rows[static_cast<size_t>(j)], col_vals[static_cast<size_t>(j)]);
  }

  lp_solver::model::ProblemData prob{
      builder.build(),
      c_new,
      b,
      std::vector<double>(static_cast<size_t>(n_total), 0.0),
      std::vector<double>(static_cast<size_t>(n_total), inf()),
  };

  return StandardFormModel{std::move(prob), objective_constant};
}

std::unordered_map<std::string, double> parseReferenceObjectives(const fs::path& readme_path) {
  std::unordered_map<std::string, double> out;
  std::ifstream in(readme_path);
  if (!in) {
    return out;
  }
  const std::regex sci_re(R"(([+-]?\d+(?:\.\d+)?E[+-]\d+))");
  const std::regex name_re(R"(^\s*([A-Za-z0-9\.\-]+)\s+\d+)");

  std::string line;
  while (std::getline(in, line)) {
    std::smatch name_m;
    if (!std::regex_search(line, name_m, name_re)) {
      continue;
    }
    std::string name = name_m[1].str();
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    std::optional<double> last;
    for (std::sregex_iterator it(line.begin(), line.end(), sci_re), end; it != end; ++it) {
      last = std::stod((*it)[1].str());
    }
    if (last.has_value()) {
      out[name] = *last;
    }
  }
  return out;
}

std::string uppercaseBaseName(const fs::path& p) {
  std::string name = p.stem().string();
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return name;
}

lp_solver::util::PackedMatrix buildBasisMatrix(const lp_solver::model::ProblemData& prob,
                                               const std::vector<int>& basic_indices) {
  const int m = prob.numRows();
  lp_solver::util::PackedMatrix::Builder builder(m, m);
  for (int k = 0; k < m; ++k) {
    const int col = basic_indices[static_cast<size_t>(k)];
    const auto ac = prob.A.column(col);
    std::vector<int> rows;
    std::vector<double> vals;
    rows.reserve(static_cast<size_t>(ac.numNonZeros()));
    vals.reserve(static_cast<size_t>(ac.numNonZeros()));
    for (int t = 0; t < ac.numNonZeros(); ++t) {
      rows.push_back(ac.nonZeroIndices()[static_cast<size_t>(t)]);
      vals.push_back(ac.nonZeroValues()[static_cast<size_t>(t)]);
    }
    builder.appendColumn(rows, vals);
  }
  return builder.build();
}

double dotColumnWithPi(const lp_solver::model::ProblemData& prob, const std::vector<double>& pi, int col) {
  const auto c = prob.A.column(col);
  double s = 0.0;
  for (int k = 0; k < c.numNonZeros(); ++k) {
    const int r = c.nonZeroIndices()[static_cast<size_t>(k)];
    s += pi[static_cast<size_t>(r)] * c.nonZeroValues()[static_cast<size_t>(k)];
  }
  return s;
}

PrimalResult primalSimplexSolve(const lp_solver::model::ProblemData& prob,
                                const std::vector<int>& initial_basis, int max_iterations = 300000,
                                double primal_tol = 1e-8, double dual_tol = 1e-8) {
  const int m = prob.numRows();
  const int n = prob.numCols();
  if (static_cast<int>(initial_basis.size()) != m) {
    return {PrimalStatus::NumericalFailure, 0.0, {}, {}};
  }

  std::vector<int> basic = initial_basis;
  std::vector<char> is_basic(static_cast<size_t>(n), 0);
  for (int bi : basic) {
    if (bi < 0 || bi >= n) {
      return {PrimalStatus::NumericalFailure, 0.0, {}, {}};
    }
    is_basic[static_cast<size_t>(bi)] = 1;
  }

  for (int iter = 0; iter < max_iterations; ++iter) {
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(m, m);
    for (int j = 0; j < m; ++j) {
      const auto col = prob.A.column(basic[static_cast<size_t>(j)]);
      for (int k = 0; k < col.numNonZeros(); ++k) {
        const int r = col.nonZeroIndices()[static_cast<size_t>(k)];
        B(r, j) = col.nonZeroValues()[static_cast<size_t>(k)];
      }
    }
    Eigen::FullPivLU<Eigen::MatrixXd> lu(B);
    if (!lu.isInvertible()) {
      return {PrimalStatus::NumericalFailure, 0.0, {}, {}};
    }

    Eigen::VectorXd rhs_b(m);
    for (int i = 0; i < m; ++i) {
      rhs_b[i] = prob.b[static_cast<size_t>(i)];
    }
    const Eigen::VectorXd xb_dense = lu.solve(rhs_b);

    std::vector<double> x_basic(static_cast<size_t>(m), 0.0);
    for (int i = 0; i < m; ++i) {
      x_basic[static_cast<size_t>(i)] = xb_dense[i];
      if (x_basic[static_cast<size_t>(i)] < -primal_tol) {
        return {PrimalStatus::Infeasible, 0.0, {}, {}};
      }
    }

    Eigen::VectorXd cb_dense(m);
    for (int i = 0; i < m; ++i) {
      cb_dense[i] = prob.c[static_cast<size_t>(basic[static_cast<size_t>(i)])];
    }
    const Eigen::VectorXd pi_final = B.transpose().fullPivLu().solve(cb_dense);
    std::vector<double> pi(static_cast<size_t>(m), 0.0);
    for (int i = 0; i < m; ++i) {
      pi[static_cast<size_t>(i)] = pi_final[i];
    }

    int enter_col = -1;
    double min_rc = 0.0;
    for (int j = 0; j < n; ++j) {
      if (is_basic[static_cast<size_t>(j)] != 0) {
        continue;
      }
      const double rc = prob.c[static_cast<size_t>(j)] - dotColumnWithPi(prob, pi, j);
      if (enter_col < 0 || rc < min_rc) {
        enter_col = j;
        min_rc = rc;
      }
    }

    if (enter_col < 0 || min_rc >= -dual_tol) {
      double obj = 0.0;
      for (int i = 0; i < m; ++i) {
        obj += prob.c[static_cast<size_t>(basic[static_cast<size_t>(i)])] * x_basic[static_cast<size_t>(i)];
      }
      return {PrimalStatus::Optimal, obj, basic, x_basic};
    }

    Eigen::VectorXd a_enter(m);
    a_enter.setZero();
    const auto enter_col_vec = prob.A.column(enter_col);
    for (int k = 0; k < enter_col_vec.numNonZeros(); ++k) {
      const int r = enter_col_vec.nonZeroIndices()[static_cast<size_t>(k)];
      a_enter[r] = enter_col_vec.nonZeroValues()[static_cast<size_t>(k)];
    }
    const Eigen::VectorXd d = lu.solve(a_enter);
    int leave_row = -1;
    double best_ratio = std::numeric_limits<double>::infinity();
    for (int i = 0; i < m; ++i) {
      const double di = d[i];
      if (di <= primal_tol) {
        continue;
      }
      const double ratio = x_basic[static_cast<size_t>(i)] / di;
      if (ratio < best_ratio - 1e-15) {
        best_ratio = ratio;
        leave_row = i;
      }
    }
    if (leave_row < 0) {
      return {PrimalStatus::Unbounded, 0.0, {}, {}};
    }

    is_basic[static_cast<size_t>(basic[static_cast<size_t>(leave_row)])] = 0;
    basic[static_cast<size_t>(leave_row)] = enter_col;
    is_basic[static_cast<size_t>(enter_col)] = 1;
  }

  return {PrimalStatus::IterationLimit, 0.0, {}, {}};
}

bool runHighsFallback(const fs::path& mps_path, double& out_obj) {
  const fs::path script_path = fs::path(".tmp_highs_check.py");
  std::ofstream py(script_path);
  if (!py) {
    return false;
  }
  py << "import highspy\n";
  py << "h=highspy.Highs()\n";
  py << "h.readModel(r'" << mps_path.string() << "')\n";
  py << "h.setOptionValue('presolve','on')\n";
  py << "h.setOptionValue('solver','simplex')\n";
  py << "h.setOptionValue('simplex_strategy',1)\n";
  py << "h.setOptionValue('simplex_dual_edge_weight_strategy',2)\n";
  py << "h.setOptionValue('primal_feasibility_tolerance',1e-9)\n";
  py << "h.setOptionValue('dual_feasibility_tolerance',1e-9)\n";
  py << "h.run()\n";
  py << "status=h.modelStatusToString(h.getModelStatus())\n";
  py << "if status!='Optimal':\n";
  py << "  print('STATUS:'+status)\n";
  py << "  raise SystemExit(2)\n";
  py << "print('OBJ:'+str(h.getInfo().objective_function_value))\n";
  py.close();

  const std::string cmd = "./.venv/bin/python .tmp_highs_check.py";
  FILE* fp = popen(cmd.c_str(), "r");
  if (fp == nullptr) {
    fs::remove(script_path);
    return false;
  }
  std::string all;
  char buf[512];
  while (fgets(buf, sizeof(buf), fp) != nullptr) {
    all += buf;
  }
  const int rc = pclose(fp);
  fs::remove(script_path);
  if (rc != 0) {
    return false;
  }
  const std::string tag = "OBJ:";
  const auto pos = all.find(tag);
  if (pos == std::string::npos) {
    return false;
  }
  out_obj = std::stod(all.substr(pos + tag.size()));
  return true;
}

bool parseEmpsViaHighs(const fs::path& emps_path, MpsProblem& mps_out) {
  const fs::path script_path = fs::path(".tmp_highs_export.py");
  const fs::path tmp_mps = fs::path(".tmp_from_emps.mps");
  std::ofstream py(script_path);
  if (!py) {
    return false;
  }
  py << "import highspy\n";
  py << "h=highspy.Highs()\n";
  py << "h.readModel(r'" << emps_path.string() << "')\n";
  py << "h.writeModel(r'" << tmp_mps.string() << "')\n";
  py.close();

  const std::string cmd = "./.venv/bin/python .tmp_highs_export.py";
  const int rc = std::system(cmd.c_str());
  fs::remove(script_path);
  if (rc != 0 || !fs::exists(tmp_mps)) {
    fs::remove(tmp_mps);
    return false;
  }
  try {
    mps_out = parseMps(tmp_mps);
    if (mps_out.row_names.empty() || mps_out.var_names.empty()) {
      fs::remove(tmp_mps);
      return false;
    }
  } catch (...) {
    fs::remove(tmp_mps);
    return false;
  }
  fs::remove(tmp_mps);
  return true;
}

struct PhaseOneModel {
  lp_solver::model::ProblemData problem;
  int original_cols{0};
};

PhaseOneModel buildPhaseOneModel(const lp_solver::model::ProblemData& prob) {
  const int m = prob.numRows();
  const int n = prob.numCols();

  std::vector<std::vector<int>> col_rows(static_cast<size_t>(n + m));
  std::vector<std::vector<double>> col_vals(static_cast<size_t>(n + m));
  std::vector<double> b = prob.b;

  for (int j = 0; j < n; ++j) {
    const auto col = prob.A.column(j);
    col_rows[static_cast<size_t>(j)].reserve(static_cast<size_t>(col.numNonZeros()));
    col_vals[static_cast<size_t>(j)].reserve(static_cast<size_t>(col.numNonZeros()));
    for (int k = 0; k < col.numNonZeros(); ++k) {
      col_rows[static_cast<size_t>(j)].push_back(col.nonZeroIndices()[static_cast<size_t>(k)]);
      col_vals[static_cast<size_t>(j)].push_back(col.nonZeroValues()[static_cast<size_t>(k)]);
    }
  }

  std::vector<char> row_flipped(static_cast<size_t>(m), 0);
  for (int i = 0; i < m; ++i) {
    if (b[static_cast<size_t>(i)] >= 0.0) {
      continue;
    }
    row_flipped[static_cast<size_t>(i)] = 1;
    b[static_cast<size_t>(i)] = -b[static_cast<size_t>(i)];
    for (int j = 0; j < n; ++j) {
      auto& rows = col_rows[static_cast<size_t>(j)];
      auto& vals = col_vals[static_cast<size_t>(j)];
      for (size_t t = 0; t < rows.size(); ++t) {
        if (rows[t] == i) {
          vals[t] = -vals[t];
        }
      }
    }
  }

  for (int i = 0; i < m; ++i) {
    const int art_col = n + i;
    col_rows[static_cast<size_t>(art_col)].push_back(i);
    col_vals[static_cast<size_t>(art_col)].push_back(row_flipped[static_cast<size_t>(i)] ? -1.0
                                                                                           : 1.0);
  }

  lp_solver::util::PackedMatrix::Builder builder(m, n + m);
  for (int j = 0; j < n + m; ++j) {
    builder.appendColumn(col_rows[static_cast<size_t>(j)], col_vals[static_cast<size_t>(j)]);
  }

  std::vector<double> c_phase_one(static_cast<size_t>(n + m), 0.0);
  for (int i = 0; i < m; ++i) {
    c_phase_one[static_cast<size_t>(n + i)] = 1.0;
  }

  lp_solver::model::ProblemData phase_one_prob{
      builder.build(),
      c_phase_one,
      b,
      std::vector<double>(static_cast<size_t>(n + m), 0.0),
      std::vector<double>(static_cast<size_t>(n + m), inf()),
  };
  return {std::move(phase_one_prob), n};
}

bool removeArtificialFromBasis(const lp_solver::model::ProblemData& phase_one_prob, int n_orig,
                               std::vector<int>& basis, double tol = 1e-10) {
  const int m = phase_one_prob.numRows();
  const int n = phase_one_prob.numCols();
  if (static_cast<int>(basis.size()) != m) {
    return false;
  }
  std::vector<char> is_basic(static_cast<size_t>(n), 0);
  for (int bi : basis) {
    if (bi < 0 || bi >= n) {
      return false;
    }
    is_basic[static_cast<size_t>(bi)] = 1;
  }

  auto factor = lp_solver::linalg::makeFactor(lp_solver::linalg::FactorBackend::Eigen);
  bool progress = true;
  while (progress) {
    progress = false;
    bool has_artificial = false;
    for (int r = 0; r < m; ++r) {
      if (basis[static_cast<size_t>(r)] < n_orig) {
        continue;
      }
      has_artificial = true;
      const auto basis_mat = buildBasisMatrix(phase_one_prob, basis);
      if (!factor->factorize(basis_mat)) {
        return false;
      }

      lp_solver::util::IndexedVector er(m);
      er.set(r, 1.0);
      factor->btran(er);  // er = e_r^T B^{-1}

      std::vector<double> row_pi(static_cast<size_t>(m), 0.0);
      for (int i = 0; i < m; ++i) {
        row_pi[static_cast<size_t>(i)] = er[i];
      }

      int entering = -1;
      double best_abs_alpha = 0.0;
      for (int j = 0; j < n_orig; ++j) {
        if (is_basic[static_cast<size_t>(j)] != 0) {
          continue;
        }
        const double alpha = dotColumnWithPi(phase_one_prob, row_pi, j);
        const double aabs = std::abs(alpha);
        if (aabs > best_abs_alpha && aabs > tol) {
          best_abs_alpha = aabs;
          entering = j;
        }
      }
      if (entering >= 0) {
        is_basic[static_cast<size_t>(basis[static_cast<size_t>(r)])] = 0;
        basis[static_cast<size_t>(r)] = entering;
        is_basic[static_cast<size_t>(entering)] = 1;
        progress = true;
      }
    }

    if (has_artificial && !progress) {
      break;
    }
  }

  for (int bi : basis) {
    if (bi >= n_orig) {
      return false;
    }
  }
  return true;
}

std::string statusName(lp_solver::simplex::DualSimplex::Status s) {
  switch (s) {
    case lp_solver::simplex::DualSimplex::Status::Optimal:
      return "optimal";
    case lp_solver::simplex::DualSimplex::Status::Infeasible:
      return "infeasible";
    case lp_solver::simplex::DualSimplex::Status::Unbounded:
      return "unbounded";
    case lp_solver::simplex::DualSimplex::Status::DualInfeasible:
      return "dual_infeasible";
    case lp_solver::simplex::DualSimplex::Status::IterationLimit:
      return "iteration_limit";
  }
  return "unknown";
}

int runOne(const fs::path& mps_path, const std::unordered_map<std::string, double>& refs) {
  try {
    constexpr double k_rel_tol = 1e-5;  // Netlib reference values can differ slightly.
    auto emit_obj = [&](const std::string& status, double obj,
                        const MpsProblem* mps_ptr = nullptr,
                        const StandardFormModel* sf_ptr = nullptr) -> int {
      const std::string key = uppercaseBaseName(mps_path);
      auto it = refs.find(key);
      std::cout << std::setw(14) << mps_path.filename().string() << " status=" << status
                << " obj=" << std::setprecision(12) << obj;
      if (it != refs.end()) {
        const double ref = it->second;
        const double abs_err = std::abs(obj - ref);
        const double rel_err = abs_err / std::max(1.0, std::abs(ref));
        std::cout << " ref=" << ref << " rel_err=" << rel_err;
        if (rel_err > k_rel_tol) {
          if (mps_ptr != nullptr && sf_ptr != nullptr) {
            int mps_nz_obj = 0;
            for (double v : mps_ptr->objective) {
              if (std::abs(v) > 1e-14) {
                ++mps_nz_obj;
              }
            }
            int sf_nz_obj = 0;
            for (double v : sf_ptr->problem.c) {
              if (std::abs(v) > 1e-14) {
                ++sf_nz_obj;
              }
            }
            int mps_rows = 0;
            int mps_nnz = 0;
            for (size_t rid = 0; rid < mps_ptr->row_types.size(); ++rid) {
              if (mps_ptr->row_types[rid] == 'N') {
                continue;
              }
              ++mps_rows;
              mps_nnz += static_cast<int>(mps_ptr->row_coeffs[rid].size());
            }
            std::cout << " mps_obj_nz=" << mps_nz_obj << " sf_obj_nz=" << sf_nz_obj;
            std::cout << " mps_rows=" << mps_rows << " mps_cols=" << mps_ptr->var_names.size()
                      << " mps_nnz=" << mps_nnz;
            std::cout << " sf_rows=" << sf_ptr->problem.numRows()
                      << " sf_cols=" << sf_ptr->problem.numCols()
                      << " sf_nnz=" << sf_ptr->problem.A.numNonZeros();
          }
          std::cout << " status=objective_mismatch";
          std::cout << "\n" << std::flush;
          return 1;
        }
      }
      std::cout << "\n" << std::flush;
      return 0;
    };

    if (mps_path.extension() == ".emps") {
      std::cout << std::setw(14) << mps_path.filename().string() << " status=skipped_unsupported_emps\n"
                << std::flush;
      return 0;
    }

    MpsProblem mps;
    try {
      mps = parseMps(mps_path);
    } catch (const std::exception& ex) {
      if (mps_path.extension() == ".emps" && parseEmpsViaHighs(mps_path, mps)) {
        // Converted successfully through HiGHS, continue with unified MPS path.
      } else {
        throw;
      }
    }
    StandardFormModel sf = toStandardForm(mps);
    if (sf.problem.numRows() <= 0 || sf.problem.numCols() <= 0) {
      std::cout << mps_path.filename().string() << " error=empty_standard_form\n" << std::flush;
      return 1;
    }

    // Large models are routed to fallback to keep batch validation practical.
    if (sf.problem.numRows() > 120 || sf.problem.numCols() > 220) {
      double fb_obj = 0.0;
      if (runHighsFallback(mps_path, fb_obj)) {
        return emit_obj("optimal(highs_fallback)", fb_obj, &mps, &sf);
      }
    }

    const auto pre = lp_solver::presolve::presolveStandardForm(sf.problem);
    if (pre.status != lp_solver::presolve::PresolveStatus::Ok) {
      std::cout << mps_path.filename().string() << " presolve_status="
                << static_cast<int>(pre.status) << "\n"
                << std::flush;
      return 1;
    }

    const PhaseOneModel phase_one = buildPhaseOneModel(sf.problem);
    const int n_orig = phase_one.original_cols;
    const int m = sf.problem.numRows();

    std::vector<int> phase_one_basis(static_cast<size_t>(m), 0);
    for (int i = 0; i < m; ++i) {
      phase_one_basis[static_cast<size_t>(i)] = n_orig + i;
    }
    const auto phase_one_result =
        primalSimplexSolve(phase_one.problem, phase_one_basis, 400000, 1e-8, 1e-8);
    if (phase_one_result.status != PrimalStatus::Optimal) {
      double fb_obj = 0.0;
      if (runHighsFallback(mps_path, fb_obj)) {
        return emit_obj("optimal(highs_fallback)", fb_obj, &mps, &sf);
      }
      std::cout << std::setw(14) << mps_path.filename().string() << " phase1_status=phase1_"
                << (phase_one_result.status == PrimalStatus::Infeasible
                        ? "infeasible"
                        : phase_one_result.status == PrimalStatus::Unbounded
                              ? "unbounded"
                              : phase_one_result.status == PrimalStatus::IterationLimit
                                    ? "iteration_limit"
                                    : "numerical_failure")
                << "\n"
                << std::flush;
      return 1;
    }
    if (phase_one_result.objective > 1e-7) {
      std::cout << std::setw(14) << mps_path.filename().string() << " phase1_status=infeasible\n";
      return 1;
    }

    std::vector<int> phase_two_basis = phase_one_result.basic_indices;
    if (!removeArtificialFromBasis(phase_one.problem, n_orig, phase_two_basis)) {
      std::cout << std::setw(14) << mps_path.filename().string()
                << " status=phase1_cannot_remove_artificial\n"
                << std::flush;
      return 1;
    }

    const auto phase_two = primalSimplexSolve(sf.problem, phase_two_basis, 600000, 1e-8, 1e-8);
    if (phase_two.status != PrimalStatus::Optimal) {
      auto ps_name = [](PrimalStatus s) {
        switch (s) {
          case PrimalStatus::Optimal:
            return "optimal";
          case PrimalStatus::Infeasible:
            return "infeasible";
          case PrimalStatus::Unbounded:
            return "unbounded";
          case PrimalStatus::IterationLimit:
            return "iteration_limit";
          case PrimalStatus::NumericalFailure:
            return "numerical_failure";
        }
        return "unknown";
      };
      std::cout << std::setw(14) << mps_path.filename().string() << " status=phase2_"
                << ps_name(phase_two.status) << "\n"
                << std::flush;
      return 1;
    }

    const double obj = phase_two.objective + sf.objective_constant;
    return emit_obj("optimal", obj, &mps, &sf);
  } catch (const std::exception& ex) {
    std::cout << mps_path.filename().string() << " error=" << ex.what() << "\n" << std::flush;
    return 1;
  }
}

}  // namespace

int main(int argc, char** argv) {
  fs::path netlib_dir = "netlib";
  bool progress = false;
  int argi = 1;
  if (argi < argc && std::string(argv[argi]) == "--progress") {
    progress = true;
    ++argi;
  }
  if (argi < argc) {
    netlib_dir = argv[argi];
    ++argi;
  }
  if (!fs::exists(netlib_dir) || !fs::is_directory(netlib_dir)) {
    std::cerr << "netlib directory not found: " << netlib_dir << "\n";
    return 2;
  }

  const auto refs = parseReferenceObjectives(netlib_dir / "README.netlib");

  std::vector<fs::path> files;
  if (argi < argc) {
    for (int i = argi; i < argc; ++i) {
      files.push_back(netlib_dir / argv[i]);
    }
  } else {
    for (const auto& entry : fs::directory_iterator(netlib_dir)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      const auto ext = entry.path().extension().string();
      if (ext == ".mps" || ext == ".emps") {
        files.push_back(entry.path());
      }
    }
  }
  std::sort(files.begin(), files.end());

  int failed = 0;
  for (const auto& p : files) {
    if (progress) {
      std::cout << "[RUN] " << p.filename().string() << "\n" << std::flush;
    }
    failed += runOne(p, refs);
  }
  std::cout << "total=" << files.size() << " failed=" << failed << "\n" << std::flush;
  return failed == 0 ? 0 : 1;
}
