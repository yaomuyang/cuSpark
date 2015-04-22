LIB=libs
SRC=src
OBJ=obj

MODULES		:= pipeline common test cuda
INNERMODULES		:= $(OBJ)/pipeline
SRC_DIR   	:= $(addprefix $(SRC)/,$(MODULES))
OBJ_DIR 	:= $(addprefix $(OBJ)/,$(MODULES))

SRCS			:= $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cc))
CUDASRCS		:= $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cu))
OBJS			:= $(patsubst $(SRC)/%.cc,$(OBJ)/%.o,$(SRCS))
OBJS		 	+= $(patsubst $(SRC)/%.cu,$(OBJ)/%.o,$(CUDASRCS))
INCLUDES	+= -I./ -I$(LIB)/include -I$(SRC)

CXX = g++ -m64
CXXFLAGS = -g -std=c++11 -O2 -L$(LIB)
LDFLAGS = -lpthread -Xlinker -rpath -Xlinker $(LIB) -L$(LIB)  -lgtest -lglog
vpath %.cc $(SRC_DIR)
vpath %.cu $(SRC_DIR)

ifeq ($(ARCH), Darwin)
# Building on mac
LDFLAGS+=-L/usr/local/cuda/lib/ -lcudart -arch=sm_21
else
# Building on Linux
LDFLAGS+=-L/usr/local/cuda/lib64/ -lcudart -arch=sm_21
endif

NVCC = nvcc -m64
NVCCFLAGS = -g -std=c++11 -O3 -L$(LIB) -rdc=true

define make-goal
ifneq ($(filter $1, $(INNERMODULES)),)
INNERFLAG := -DINNERFLAG
endif
$1/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INNERFLAG) $(INCLUDES) -c $$< -o $$@
$1/%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INNERFLAG) $(INCLUDES) -c $$< -o $$@
endef

.PHONY: all test checkdirs clean

all: checkdirs testall

testall: $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(INCLUDES) -o testall $(SRC)/testall/testall.cc $(OBJS)

print-%  : ; @echo $* = $($*)

checkdirs: $(OBJ_DIR)

$(OBJ_DIR):
	@mkdir -p $@

clean:
	rm -r obj testall

$(foreach odir,$(OBJ_DIR),$(eval $(call make-goal,$(odir))))
