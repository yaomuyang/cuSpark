#ifndef CUSPARK_PIPELINE_PIPELINE_H_
#define CUSPARK_PIPELINE_PIPELINE_H_

#include <iostream>
#include <fstream>
#include <string>
#include "common/types.h"
#include "common/logging.h"
#include "pipeline/pipeline.h"
#include "cuda/cuda-basics.h"

namespace cuspark {

template <typename T>
class PipeLine;

/*
 * Basic PipeLine class, which we generate from file or array
 * 
 */
class StringPipeLine {
  public:
    StringPipeLine(char* fileName, uint32_t size)
	:size_(size){
      if(typeid(T)!=typeid(std::string)){
        DLOG(ERROR) << "PipeLine have to be typed as <string>, if you hope to initiate this from file";
	exit(1);
      }
      data_ = new std::string[size_];

      std::ifstream infile(fileName);
      std::string line;
      int line_number = 0;
      while(std::getline(infile, line)){
	data_[line_number++] = line;
      }
    }

    template <typename T>
    PipeLine<T> StringMap(StringMapFunction<T> f){
      PipeLine<T> a(this, f);
      return a;
    }
    
    uint32_t GetDataSize(){
	return size_;
    }
    
    void Execute(){
      DLOG(INFO) << "Executing PipeLine";
    }

    uint32_t size_; //the length of the data array
    T* data_; //pointer to the array

};

}

#endif // CUSPARK_SRC_PIPELINE_PIPELINE_H_
