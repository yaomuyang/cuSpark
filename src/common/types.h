#ifndef CUSPARK_COMMON_TYPES_H
#define CUSPARK_COMMON_TYPES_H

#include <boost/function.hpp>
#include <boost/function_equal.hpp>

namespace cuspark {
 
  template <typename T, typename U, typename F>
  class MapFunction{
    public:
      const F f;
      __host__ __device__ MapFunction(F f) : f(f){}
      __host__ __device__ U operator()(T arg) const { return f(arg); }
  };
  //using MapFunction = std::function<U(T)>;

  template <typename T>
  using ReduceFunction = boost::function<T (const T& a, const T& b)>;


}



#endif
