# MAT3007H LP Solver (Revised Dual Simplex)

C++20 implementation of a **revised dual simplex** engine aligned with
[mat3007h_Project_Manual.tex](mat3007h_Project_Manual.tex): sparse LU (Eigen), ETA / PFI updates,
Harris ratio test, optional dual steepest edge with Goldfarb–Reid weights, Gilbert–Peierls
reachability helpers, and lightweight presolve checks.

## Layout

| Area | Role |
|------|------|
| `include/lp_solver/util/` | `PackedMatrix` (CSC), `IndexedVector` |
| `include/lp_solver/linalg/` | `IBasisFactor`, `EigenFactor` / `UmfpackFactor`, `gilbert_peierls.hpp` |
| `include/lp_solver/model/` | `ProblemData`, `SolverState`, `initialization.hpp` |
| `include/lp_solver/simplex/` | `DualSimplex`, `NaiveRowPivot`, `DseRowPivot`, observers |
| `include/lp_solver/presolve/` | `presolver.hpp` |

`FactorBackend::Umfpack` currently aliases the same Eigen SparseLU core (no external SuiteSparse
required) so CI and coursework builds stay self-contained.

## Dependencies

- **CMake** ≥ 3.20, **Ninja** recommended
- **Eigen 3.4** fetched automatically via `FetchContent` (header-only extract)
- **GoogleTest** for unit tests (also via `FetchContent`)

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

## Stress factor micro-benchmark

```bash
cmake --build build --target lp_solver_stress
./build/lp_solver_stress --dim=2000 --iters=5000 --seed=42 --csv=build/stress_factor_results.csv
```

`lp_solver_stress_smoke` runs a short regression via CTest.

## Manual cross-reference

| Manual section | Code |
|----------------|------|
| CSC / `IndexedVector` | `util/` |
| Sparse LU + ETA | `linalg/eigen_factor.*` |
| CHUZR / BTRAN / CHUZC / FTRAN | `simplex/dual_simplex.cpp` |
| Harris two-pass | `DualSimplex::chuzc` when `SolverConfig::use_harris_ratio_test` |
| Dual steepest edge + GR | `DseRowPivot`, `goldfarbReidUpdate` |
| Gilbert–Peierls reachability | `linalg/gilbert_peierls.*` |
| Presolve (empty row/column checks) | `presolve/presolver.cpp` |
| Phase I (dual infeasibility flag) | `model/phase_one.hpp` (`hasNegativeReducedCost`) |

Automatic Big‑M row/column augmentation is not wired into `DualSimplex` yet; models that need it
should extend `ProblemData` manually (see the manual’s bounding constraint) before calling
`initializeSlackBasis` / `DualSimplex::solve`.

## Cursor / VS Code

Use `.vscode/tasks.json` and `.vscode/launch.json`: run the **Configure Debug** task once, then `F5`
(`Debug lp_solver_tests`).
