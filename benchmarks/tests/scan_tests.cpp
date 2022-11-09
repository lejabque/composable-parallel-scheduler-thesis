#include "scan.h"
#include <gtest/gtest.h>

TEST(ParallelFor, PrefixSum) {
  std::vector<uint64_t> data(1 << 10);
  for (size_t i = 0; i != data.size(); ++i) {
    data[i] = i + 1;
  }
  Scan::Scan(10, data);
  uint64_t sum = 0;
  for (size_t i = 0; i != data.size(); ++i) {
    sum += i;
    EXPECT_EQ(data[i], sum);
  }
}
