#ifdef TBB_MODE
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif
#include <cassert>
#include <cstddef>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace SPMV {

struct MatrixDimensions {
  size_t Rows;
  size_t Columns;
};

struct DenseMatrix {
  MatrixDimensions Dimensions;
  std::vector<std::vector<double>> Data;
};

struct SparseMatrixCSR {
  MatrixDimensions Dimensions;
  std::vector<double> Values;
  std::vector<size_t> ColumnIndex;
  std::vector<size_t> RowIndex;
};

inline double MultiplyRow(const SPMV::SparseMatrixCSR &A,
                          const std::vector<double> &x, size_t row) {
  double y = 0;
  for (size_t i = A.RowIndex[row]; i < A.RowIndex[row + 1]; ++i) {
    y += A.Values[i] * x[A.ColumnIndex[i]];
  }
  return y;
}

SparseMatrixCSR DenseToSparse(const DenseMatrix &in);

void DensePmvSerial(const DenseMatrix &A, const std::vector<double> &x,
                    std::vector<double> &y);

enum class SparseKind {
  BALANCED,
  UNBALANCED,
};

inline thread_local std::mt19937_64 RandomGenerator{std::random_device()()};

template <SparseKind Kind>
static SparseMatrixCSR GenSparseMatrix(size_t n, size_t m, double density) {
  assert(0 <= density && density <= 1.0);

  SparseMatrixCSR out;
  out.Dimensions.Rows = n;
  out.Dimensions.Columns = m;
  auto densityGen = std::bernoulli_distribution(density);
  auto valueGen = std::uniform_real_distribution<double>(-1e9, 1e9);

  std::set<std::pair<size_t, size_t>> positions;
  size_t elements_n = n * m * density;
  while (positions.size() < elements_n) {
    if constexpr (Kind == SparseKind::BALANCED) {
      positions.insert({RandomGenerator() % n, RandomGenerator() % m});
    } else if constexpr (Kind == SparseKind::UNBALANCED) {
      auto row = RandomGenerator() % n;
      auto col = RandomGenerator() % row;
      positions.insert({row, col});
    } else {
      // unreachable
      static_assert(Kind == SparseKind::BALANCED ||
                    Kind == SparseKind::UNBALANCED);
    }
  }

  out.RowIndex.reserve(n + 1);
  out.RowIndex.push_back(0);
  for (auto it = positions.begin(); it != positions.end(); ++it) {
    auto [i, j] = *it;
    while (i > out.RowIndex.size() - 1) {
      out.RowIndex.push_back(out.ColumnIndex.size());
    }
    out.Values.push_back(valueGen(RandomGenerator));
    out.ColumnIndex.push_back(j);
  }
  while (out.RowIndex.size() < n + 1) {
    out.RowIndex.push_back(out.ColumnIndex.size());
  }

  return out;
}

static std::vector<double> GenVector(size_t m) {
  auto valueGen = std::uniform_real_distribution<double>(-1e9, 1e9);
  std::vector<double> x;
  x.resize(m);
  for (auto &el : x) {
    el = valueGen(RandomGenerator);
  }
  return x;
}
} // namespace SPMV
