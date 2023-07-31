#pragma once

#include "../parallel_for.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <algorithm>
#include <type_traits>
#include <utility>
#include <functional>
#include <vector>

#include <mutex>

namespace SPMV {

inline constexpr size_t STEP = 9;

struct MatrixDimensions {
  size_t Rows;
  size_t Columns;
};

template <typename T> struct DenseMatrix {
  DenseMatrix(size_t n, size_t m)
      : Dimensions{n, m}, Data(n, std::vector<T>(m)) {}

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
void __attribute__((noinline,noipa)) MultiplyMatrix(const SPMV::SparseMatrixCSR<T> &A, const std::vector<T> &x,
                    std::vector<T> &out, size_t grainSize = 1) {
  assert(A.Dimensions.Columns == x.size());
  ParallelFor(
      0, A.Dimensions.Rows, [&](size_t i) { out[i] = MultiplyRow(A, x, i); },
      grainSize);
}

template <typename T>
void __attribute__((noinline,noipa)) MultiplyMatrix(const SPMV::DenseMatrix<T> &A, const std::vector<T> &x,
                    std::vector<T> &out, size_t grainSize = 1) {
  ParallelFor(
      0, A.Dimensions.Rows,
      [&](size_t i) {
        out[i] = 0;
        for (size_t j = 0; j != A.Dimensions.Columns; ++j) {
          out[i] += A.Data[i][j] * x[j];
        }
      },
      grainSize);
}
template <typename T>
void MultiplyMatrix(const SPMV::DenseMatrix<T> &A,
                    const SPMV::DenseMatrix<T> &B, SPMV::DenseMatrix<T> &out,
                    size_t grainSize = 1) {
  ParallelFor(
      0, out.Dimensions.Rows,
      [&](size_t row) {
        ParallelFor(
            0, out.Dimensions.Columns,
            [&](size_t col) {
              T sum{};
              for (size_t j = 0; j != A.Dimensions.Columns; ++j) {
                sum += A.Data[row][j] * B.Data[j][col];
              }
              out.Data[row][col] = sum;
            },
            grainSize);
      },
      grainSize);
}

template <typename T>
void __attribute__((noinline,noipa)) TransposeMatrix(SPMV::DenseMatrix<T> &input, SPMV::DenseMatrix<T> &out,
                     size_t blocks = 16, size_t grainSize = 1) {
  assert(input.Dimensions.Rows == out.Dimensions.Columns);
  assert(input.Dimensions.Columns == out.Dimensions.Rows);
  auto blocksRows = std::min(blocks, input.Dimensions.Rows);
  auto blocksColumns = std::min(blocks, input.Dimensions.Columns);
  auto blockRowSize = (input.Dimensions.Rows + blocksRows - 1) / blocksRows;
  auto blockColumnSize =
      (input.Dimensions.Columns + blocksColumns - 1) / blocksColumns;
  ParallelFor(
      0, blocksRows,
      [&](size_t row) {
        ParallelFor(0, blocksColumns, [&](size_t column) {
          auto fromRow = row * blockRowSize;
          auto fromCol = column * blockColumnSize;
          for (size_t i = fromRow;
               i != std::min(input.Dimensions.Rows, fromRow + blockRowSize);
               ++i) {
            for (size_t j = fromCol; j != std::min(input.Dimensions.Columns,
                                                   fromCol + blockColumnSize);
                 ++j) {
              out.Data[j][i] = input.Data[i][j];
            }
          }
        });
      },
      grainSize);
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
  TRIANGLE,
  HYPERBOLIC,
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

  std::vector<std::pair<size_t, size_t>> positions;
  size_t elements_n = n * m * density;
  positions.reserve(elements_n + 100);
  if constexpr (Kind == SparseKind::BALANCED) {
    auto posGen = std::uniform_int_distribution<size_t>(0, STEP - 1);
    for (size_t i = 0; i != n; ++i) {
      for (size_t j = 0; j < m; j += STEP) {
        size_t colOffset = posGen(RandomGenerator);
        positions.push_back({i, std::min(j + colOffset, m - 1)});
      }
    }
  } else if constexpr (Kind == SparseKind::HYPERBOLIC) {
    for (size_t i = 0; i != n; ++i) {
      // sum of elementsCount = density * n * m / std::log(n + 1) * (1 + 1/2 +
      // 1/3 + ... + 1/n) ~ density * n * m
      size_t elementsCount = density * n * m / std::log(n + 1) / (i + 1);
      if (elementsCount == 0) {
        continue;
      }
      size_t step = std::max(m / elementsCount, 1ul);
      auto posGen = std::uniform_int_distribution<size_t>(0, step - 1);
      for (size_t j = 0; j < m; j += step) {
        size_t pos = posGen(RandomGenerator);
        positions.push_back({i, std::min(j + pos, m - 1)});
      }
    }
  } else if constexpr (Kind == SparseKind::TRIANGLE) {
    for (size_t i = 0; i != n; ++i) {
      // distribute elements like triangle
      // sum of elementsCountInRow = density * m * 2 /n * (1 + 2 + 3 + ... + n)
      // ~ density * m * (n + 1)
      size_t width = std::min(i, m - 1);
      size_t elementsCountInRow = density * m * (n - i) * 2 / n;
      if (elementsCountInRow == 0) {
        continue;
      }
      size_t step = std::max(width / elementsCountInRow, 1ul);
      auto posGen = std::uniform_int_distribution<size_t>(0, step - 1);
      for (size_t j = 0; j <= width; j += step) {
        size_t pos = posGen(RandomGenerator);
        positions.push_back({i, std::min(j + pos, width)});
      }
    }
  } else {
    // check that all possible kinds are handled
    static_assert(Kind == SparseKind::BALANCED ||
                  Kind == SparseKind::HYPERBOLIC ||
                  Kind == SparseKind::TRIANGLE);
  }


  std::sort(positions.begin(), positions.end());
  out.RowIndex.reserve(n + 20);
  out.RowIndex.push_back(0);
  out.Values.reserve(1000 + density * n * m);
  out.ColumnIndex.reserve(1000 + density * n * m);
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

template <typename T> DenseMatrix<T> GenDenseMatrix(size_t rows, size_t cols) {
  DenseMatrix<T> out(rows, cols);
  auto valueGen = []() {
    if constexpr (std::is_floating_point_v<T>) {
      return std::uniform_real_distribution<T>(-1e9, 1e9);
    } else {
      return std::uniform_int_distribution<T>(-1e9, 1e9);
    }
  }();

  for (auto &row : out.Data) {
    for (auto &el : row) {
      el = valueGen(RandomGenerator);
    }
  }
  return out;
}

inline const size_t MATRIX_SIZE = (GetNumThreads() << 9) + (GetNumThreads() << 4) + 7;
inline constexpr double DENSITY = 1.0 / STEP;
} // namespace SPMV
