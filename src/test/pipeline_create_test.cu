#include <gtest/gtest.h>
#include "pipeline/pipeline.h"

using namespace cuspark;

class PipeLineCreateTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};


TEST_F(PipeLineCreateTest, Basic) {
  uint32_t N = 5;

  int data[N];
  uint32_t i;

  for (i = 0; i < N; ++i) {
    data[i] = i;
  }

  PipeLine<int> pl(data, N);

  EXPECT_EQ(N, pl.GetDataSize());

  int* out = pl.GetData();
  for (i = 0; i < N; ++i) {
    EXPECT_EQ(out[i], data[i]);
  }

  /*
  for (i = 0; i < N; ++i) {
    EXPECT_EQ(pl.GetElement(i), data[i]);
  }
  */
}

