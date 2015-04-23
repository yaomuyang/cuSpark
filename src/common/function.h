#ifndef CUSPARK_COMMON_TYPES_H
#define CUSPARK_COMMON_TYPES_H

#include <stdio.h>
#include <sstream>
#include <string>
#include <common/logging.h>

namespace cuspark {
 
  struct point{
    float4 x;
    double y;
  };

  template<typename T, typename U>
  struct MapFunction {
    float4 w_;
    MapFunction(float4 w) : w_(w) {}
    __host__ __device__ float4 operator()(point arg) { 
      float dotproduct = arg.x.x * w_.x + arg.x.y * w_.y + arg.x.z * w_.z + arg.x.w * w_.w;
      dotproduct = (1/(1+exp(-arg.y * dotproduct)) - 1) * arg.y;
      return make_float4(arg.x.x*dotproduct, arg.x.y*dotproduct, arg.x.z*dotproduct, arg.x.w*dotproduct);
    }
  };

  template<typename T>
  struct StringMapFunction {
    point operator()(std::string arg) {
      std::stringstream iss(arg);
      point p;
      iss >> p.x.x  >> p.x.y  >> p.x.z >> p.x.w >> p.y;
      return p;
    }
  };

  template <typename T>
  struct ReduceFunction { 
    __host__ __device__ float4 operator()(float4 arg1, float4 arg2){
      return make_float4(arg1.x+arg2.x, arg1.y+arg2.y, arg1.z+arg2.z, arg1.w+arg2.w);
    }
  };

}



#endif
