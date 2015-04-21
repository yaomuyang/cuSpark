#ifndef CUSPARK_PIPELINE_MAPPEDPIPELINE_H_
#define CUSPARK_PIPELINE_MAPPEDPIPELINE_H_

#include <stdio.h>
#include "common/types.h"
#include "common/logging.h"
#include "common/function.h"
#include "cuda/cuda-basics.h"
#include "pipeline/pipeline.h"

namespace cuspark {

template <typename T, typename U>
__global__ void map_kernel(T* input, U* output, int size){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < size){
    output[i] = map(input[i]);
  }
}

/**
 * Mapped from type U to type T
 */
template <typename T, typename U>
class MappedPipeLine : public PipeLine<T> {
  public:
    MappedPipeLine(PipeLine<U> *parent)
        : PipeLine<T>(parent->GetDataSize()),
	  parent_(parent) {}

    template <typename W>
    MappedPipeLine<T, W> Map(MapFunction<T, W, U(*)(T)> f);

    void Execute(){
      parent_ -> Execute();
      DLOG(INFO) << "Executing MappedPipeLine";
      PipeLine<T>::MallocData();
      int num_blocks = (this->size_ + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      map_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(parent_ -> data_, this->data_, this->size_);
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

    //MapFunction<U, T, U(*)(T)> f_;
    PipeLine<U> *parent_;

};

}

#endif //CUSPARK_SRC_PIPELINE_MAPPEDPIPELINE_H_
