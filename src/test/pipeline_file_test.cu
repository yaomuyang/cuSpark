#include <iostream>
#include <gtest/gtest.h>
#include <common/function.h>
#include "pipeline/pipeline.h"
#include "pipeline/mappedpipeline.h"

using namespace cuspark;

class PipeLineFileTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(PipeLineFileTest, Basic) {

  StringMapFunction<point> f1;
  PipeLine<point> pl("/afs/andrew.cmu.edu/usr6/muyangya/private/15418/project/cuSpark/dataset.txt", 32928, f1);
 
  EXPECT_EQ(32928, pl.GetDataSize());

  point* out = pl.GetData();
  EXPECT_EQ(out[484].x1, 4.7432);
  EXPECT_EQ(out[484].y, 0);
  EXPECT_EQ(out[31003].x2, -5.0966);
  EXPECT_EQ(out[32927].x3, 2.6842);
  
}

