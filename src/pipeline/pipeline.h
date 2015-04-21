#ifndef CUSPARK_PIPELINE_PIPELINE_H_
#define CUSPARK_PIPELINE_PIPELINE_H_

#include "common/types.h"
#include "common/logging.h"
#include "pipeline/pipeline.h"
#include "cuda/cuda-basics.h"

namespace cuspark {

template <typename T, typename U>
class MappedPipeLine;

template <typename T>
class PipeLine {
  public:
    PipeLine(T *data, uint32_t size)
	:size_(size){
      DLOG(INFO) << "initiating GPU memory for data with size :" << sizeof(T) << " * " << size;
      MallocData();
      cudaMemcpy(data_, data, size_ * sizeof(T), cudaMemcpyHostToDevice);
    }

    PipeLine(uint32_t size)
  	: size_(size){}
   
    template <typename U>
    MappedPipeLine<U, T> Map(MapFunction<U, T> f){
      MappedPipeLine<U, T> a(this, f);
      return a;
    }
    
    T Reduce(ReduceFunction<T> f);
  
    uint32_t GetDataSize(){
	return size_;
    }
    
    void MallocData(){
      cudaMalloc((void**)&data_, size_ * sizeof(T));
    }
 
    void FreeData(){
      cudaFree(data_);
    }
 
    T *GetData(){
      DLOG(INFO) << "Getting data from address: " << data_;
      T* data = (T*)malloc(size_ * sizeof(T));
      cudaMemcpy(data, this->data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
      return data;
    }

    T GetElement(uint32_t index){
      T element;
      cudaMemcpy(&element, this->data_ + index, sizeof(T), cudaMemcpyDeviceToHost);
      return element;
    }

    void Execute(){
      DLOG(INFO) << "Executing PipeLine";
    }

    uint32_t size_; //the length of the data array
    T* data_; //pointer to the array

};

}

#endif // CUSPARK_SRC_PIPELINE_PIPELINE_H_
