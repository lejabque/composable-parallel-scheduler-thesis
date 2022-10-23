#include "spmv.h"
#include <iostream>
#include <thread>

SPMV::SparseMatrixCSR SPMV::DenseToSparse(const DenseMatrix &in) {
  SparseMatrixCSR out;
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

void SPMV::DensePmvSerial(const DenseMatrix &A, const std::vector<double> &x,
                          std::vector<double> &y) {
  y.resize(A.Dimensions.Rows);
  for (size_t i = 0; i < A.Dimensions.Rows; ++i) {
    for (size_t j = 0; j < A.Dimensions.Columns; ++j) {
      y[i] += A.Data[i][j] * x[j];
    }
  }
}
