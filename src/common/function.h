#ifndef CUSPARK_COMMON_TYPES_H
#define CUSPARK_COMMON_TYPES_H

#include <sstream>
#include <string>
#include <common/logging.h>
#include <boost/function.hpp>
#include <boost/function_equal.hpp>

namespace cuspark {
  
  struct pair_struct{
    int a1, a2;
  };

  struct point{
    double x1, x2, x3, x4, y;
  };

  template<typename T, typename U>
  struct MapFunction {
    int x_, y_;
    MapFunction(int x, int y) : x_(x), y_(y){}
    __host__ __device__ pair_struct operator()(int arg) { 
      pair_struct t;
      t.a1 = x_ * arg;
      t.a2 = y_ + arg * arg;
      return t;
    }
  };

  template<typename T>
  struct StringMapFunction {
    point operator()(std::string arg) {
      std::stringstream iss(arg);
      point p;
      iss >> p.x1  >> p.x2  >> p.x3 >> p.x4 >> p.y;
      return p;
    }
  };

  template <typename T>
  using ReduceFunction = boost::function<T (const T& a, const T& b)>;


}



#endif
