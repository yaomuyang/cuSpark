#include <iostream>
#include <gtest/gtest.h>
#include <common/types.h>
#include "pipeline/pipeline.h"
#include "pipeline/mappedpipeline.h"

using namespace cuspark;

class PipeLineMapTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(PipeLineMapTest, Basic) {
  uint32_t N = 10000;

  int data[N];
  uint32_t i;

  for (i = 0; i < N; ++i) {
    data[i] = i;
  }

  PipeLine<int> pl(data, N);
 
  int constant1 = 7;
  int constant2 = 4; 
  MapFunction<pair_struct, int> f{constant1, constant2};
  MappedPipeLine<pair_struct, int> mpl = pl.Map(f);

  EXPECT_EQ(N, mpl.GetDataSize());

  pair_struct* out = mpl.GetData();
  for (i = 0; i < N; ++i) {
    EXPECT_EQ(out[i].a1, constant1 * data[i]);
    EXPECT_EQ(out[i].a2, constant2 + data[i] * data[i]);
  }
  
}

