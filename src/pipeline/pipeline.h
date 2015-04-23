#ifndef CUSPARK_PIPELINE_PIPELINE_H_
#define CUSPARK_PIPELINE_PIPELINE_H_

#include <iostream>
#include <fstream>
#include <string>
#include "common/function.h"
#include "common/logging.h"
#include "cuda/cuda-basics.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>

namespace cuspark {

template <typename T, typename U>
class MappedPipeLine;

/*
 * Basic PipeLine class, which we generate from file or array
 * 
 */
template <typename T>
class PipeLine {
  public:
    PipeLine(T *data, uint32_t size):size_(size){
      DLOG(INFO) << "initiating pipeline from array";
      MallocCudaData();
      cudaMemcpy(data_, data, size_ * sizeof(T), cudaMemcpyHostToDevice);
    }

    PipeLine(std::string filename, uint32_t size, StringMapFunction<T> f):size_(size){
      DLOG(INFO) << "initiating pipeline from file: "<<size_;
      MallocCudaData();
      T* cache = new T[size_];

      std::ifstream infile;
      infile.open(filename);
      std::string line;
      int line_number = 0;
      while(std::getline(infile, line)){
        cache[line_number++] = f(line);
      }
      cudaMemcpy(data_, cache, size_ * sizeof(T), cudaMemcpyHostToDevice);
      free(cache);
    }

    PipeLine(uint32_t size):size_(size){}
   
    template <typename U>
    MappedPipeLine<U, T> Map(MapFunction<U, T> f){
      MappedPipeLine<U, T> a(this, f);
      return a;
    }
    
    T Reduce(ReduceFunction<T> f){
      DLOG(INFO) << "Executing Reduce";
      thrust::device_ptr<T> self_data(data_);
      T init = GetElement_(0);
      T result = thrust::reduce(self_data + 1, self_data + size_, init, f);
      FreeCudaData();
      return result;
    }
  
    uint32_t GetDataSize(){
	return size_;
    }
    
    T *GetData(){
      Execute();
      return GetData_();
    }
 
    T *GetData_(){
      DLOG(INFO) << "Getting data from address: " << data_;
      T* data = (T*)malloc(size_ * sizeof(T));
      cudaMemcpy(data, this->data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
      return data;
    }

    T GetElement(uint32_t index){
      Execute();
      return GetElement_(index);
    }

    T GetElement_(uint32_t index){
      T element;
      cudaMemcpy(&element, this->data_ + index, sizeof(T), cudaMemcpyDeviceToHost);
      return element;
    }

    void Cache(){
      cached = true;
    }

  //protected:

    T* data_; //pointer to the array
    bool cached = false;
    uint32_t size_; //the length of the data array

    void MallocCudaData(){
      DLOG(INFO) << "malloc GPU memory for data with size :" << sizeof(T) << " * " << size_;
      cudaMalloc((void**)&data_, size_ * sizeof(T));
    }
 
    void FreeCudaData(){
      if(!cached){
        DLOG(INFO) << "freeing GPU memory for data with size :" << sizeof(T) << " * " << size_;
	cudaFree(data_);
        data_ = NULL;
      }
    }

    void Execute(){
      DLOG(INFO) << "Executing PipeLine";
    }

};

}

#endif // CUSPARK_SRC_PIPELINE_PIPELINE_H_
