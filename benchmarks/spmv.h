#include "../include/parallel_for.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

namespace SPMV {

struct MatrixDimensions {
  size_t Rows;
  size_t Columns;
};

template <typename T> struct DenseMatrix {
  MatrixDimensions Dimensions;
  std::vector<std::vector<T>> Data;
};

template <typename T> struct SparseMatrixCSR {
  MatrixDimensions Dimensions;
  std::vector<T> Values;
  std::vector<size_t> ColumnIndex;
  std::vector<size_t> RowIndex;
};

template <typename T>
T MultiplyRow(const SPMV::SparseMatrixCSR<T> &A, const std::vector<T> &x,
              size_t row) {
  T y = 0;
  for (size_t i = A.RowIndex[row]; i < A.RowIndex[row + 1]; ++i) {
    y += A.Values[i] * x[A.ColumnIndex[i]];
  }
  return y;
}

template <typename T>
void MultiplyMatrix(const SPMV::SparseMatrixCSR<T> &A, const std::vector<T> &x,
                    std::vector<T> &out, size_t grainSize = 1) {
  ParallelFor(0, A.Dimensions.Rows,
              [&](size_t i) { out[i] = MultiplyRow(A, x, i); }, grainSize);
}

template <typename T>
void MultiplyMatrix(const SPMV::DenseMatrix<T> &A, const std::vector<T> &x,
                    std::vector<T> &out, size_t grainSize = 1) {
  ParallelFor(0, A.Dimensions.Rows, [&](size_t i) {
    out[i] = 0;
    for (size_t j = 0; j != A.Dimensions.Columns; ++j) {
      out[i] += A.Data[i][j] * x[j];
    }
  }, grainSize);
}

template <typename T>
SparseMatrixCSR<T> DenseToSparse(const DenseMatrix<T> &in) {
  SparseMatrixCSR<T> out;
  out.Dimensions = in.Dimensions;
  out.RowIndex.push_back(0);
  for (size_t i = 0; i < in.Dimensions.Rows; ++i) {
    for (size_t j = 0; j < in.Dimensions.Columns; ++j) {
      if (in.Data[i][j] != 0) {
        out.Values.push_back(in.Data[i][j]);
        out.ColumnIndex.push_back(j);
      }
    }
    out.RowIndex.push_back(out.ColumnIndex.size());
  }
  return out;
}

template <typename T>
void DensePmvSerial(const DenseMatrix<T> &A, const std::vector<T> &x,
                    std::vector<T> &y) {
  y.resize(A.Dimensions.Rows);
  for (size_t i = 0; i < A.Dimensions.Rows; ++i) {
    for (size_t j = 0; j < A.Dimensions.Columns; ++j) {
      y[i] += A.Data[i][j] * x[j];
    }
  }
}

enum class SparseKind {
  BALANCED,
  UNBALANCED,
};

inline thread_local std::default_random_engine RandomGenerator{
    std::random_device()()};

template <typename T, SparseKind Kind>
SparseMatrixCSR<T> GenSparseMatrix(size_t n, size_t m, double density) {
  assert(0 <= density && density <= 1.0);

  SparseMatrixCSR<T> out;
  out.Dimensions.Rows = n;
  out.Dimensions.Columns = m;
  auto valueGen = std::uniform_real_distribution<T>(-1e9, 1e9);

  std::set<std::pair<size_t, size_t>> positions;
  size_t elements_n = n * m * density;
  if constexpr (Kind == SparseKind::BALANCED) {
    auto posGen = std::uniform_int_distribution<size_t>(0, n * m - 1);
    for (size_t i = 0; i < n * m * density; ++i) {
      size_t pos = posGen(RandomGenerator);
      positions.insert({pos / m, pos % m});
    }
  } else if constexpr (Kind == SparseKind::UNBALANCED) {
    auto posGen = std::uniform_int_distribution<size_t>(0, m - 1);
    for (size_t i = 0; i != n; ++i) {
      // sum of elementsCount = density * n * m / std::log(n + 1) * (1 + 1/2 +
      // 1/3 + ... + 1/n) ~ density * n * m
      size_t elementsCount = density * n * m / std::log(n + 1) / (i + 1);
      for (size_t j = 0; j != elementsCount; ++j) {
        size_t pos = posGen(RandomGenerator);
        positions.insert({i, pos});
      }
    }
  } else {
    // check that all possible kinds are handled
    static_assert(Kind == SparseKind::BALANCED ||
                  Kind == SparseKind::UNBALANCED);
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

template <typename T> std::vector<T> GenVector(size_t m) {
  std::vector<T> x(m);
  if constexpr (std::is_floating_point_v<T>) {
    auto valueGen = std::uniform_real_distribution<T>(-1e9, 1e9);
    for (auto &el : x) {
      el = valueGen(RandomGenerator);
    }
  } else {
    auto valueGen = std::uniform_int_distribution<T>(0, 1e6);
    for (auto &el : x) {
      el = valueGen(RandomGenerator);
    }
  }
  return x;
}
} // namespace SPMV
