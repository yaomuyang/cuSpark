#ifndef CUSPARK_PIPELINE_MAPPEDPIPELINE_H_
#define CUSPARK_PIPELINE_MAPPEDPIPELINE_H_

#include <stdio.h>
#include "common/function.h"
#include "common/logging.h"
#include "cuda/cuda-basics.h"
#include "pipeline/pipeline.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>

namespace cuspark {

template <typename T, typename U>
__global__ void map_kernel(U* input, T* output, int size, MapFunction<T, U> map){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < size){
    output[i] = map(input[i]);
  }
}

/*
 * Mapped from type U to type T
 */
template <typename T, typename U>
class MappedPipeLine : public PipeLine<T> {
  public:
    MappedPipeLine(PipeLine<U> *parent, MapFunction<T, U> f)
        : PipeLine<T>(parent->GetDataSize()),
	  parent_(parent),
          f_(f) {}

    template <typename W>
    MappedPipeLine<W, T> Map(MapFunction<W, T> f);

    void Execute(){
      parent_ -> Execute();
      DLOG(INFO) << "Executing MappedPipeLine";
      PipeLine<T>::MallocCudaData();
      thrust::device_ptr<U> parent_data(parent_ -> data_);
      thrust::device_ptr<T> child_data(this -> data_);
      thrust::transform(parent_data, parent_data + this->size_ * sizeof(T), child_data, f_);
      //int num_blocks = (this->size_ + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      //map_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(parent_ -> data_, this->data_, this->size_, f_);
      cudaThreadSynchronize();
    }

    T Reduce(ReduceFunction<T> f);
    
    T *GetData() {
      Execute();
      return PipeLine<T>::GetData(); 
    }

    T GetElement(uint32_t index) {
      Execute();
      return PipeLine<T>::GetElement(index);
    }

  protected:

    MapFunction<T, U> f_;
    PipeLine<U> *parent_;

};

}

#endif //CUSPARK_SRC_PIPELINE_MAPPEDPIPELINE_H_
