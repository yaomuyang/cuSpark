#include <gtest/gtest.h>
#include <common/function.h>
#include "pipeline/pipeline.h"
#include "pipeline/mappedpipeline.h"

using namespace cuspark;

class PipeLineLogisticRegressionTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TEST_F(PipeLineLogisticRegressionTest, Basic) {

  float eta = 0.01;

  StringMapFunction<point> f1;
  PipeLine<point> pl("/tmp/muyangya/lrdata.txt", 1251264, f1);
  //PipeLine<point> pl("/tmp/muyangya/lrdatasmall.txt", 32928, f1);
  pl.Cache();
  point* out = pl.GetData();

  float4 w = make_float4(1,1,1,1);
  for(int i = 0; i < 1000; i++){
    MapFunction<float4, point> map{w};
    ReduceFunction<float4> reduce;
    MappedPipeLine<float4, point> mpl = pl.Map(map);
    float4 wdiff = mpl.Reduce(reduce);
    w = make_float4(w.x+eta*wdiff.x, w.y+eta*wdiff.y, w.z+eta*wdiff.z, w.w+eta*wdiff.w);
    std::cout<<"[DEBUG] iteration: #"<< i << ", wdiff:" <<wdiff.x<<", "<<wdiff.y<<", "<<wdiff.z<<", "<<wdiff.w<<std::endl;
  }
}

