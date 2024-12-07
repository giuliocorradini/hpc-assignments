ifndef CUDA_HOME
CUDA_HOME:=/usr/local/cuda
endif

ifndef VERSION
VERSION=gramschmidt.cu
endif

BUILD_DIR ?= ./build

NVCC=$(CUDA_HOME)/bin/nvcc
CXX=g++

OPT:=-O2 -g
NVOPT:=-Xcompiler -fopenmp -lineinfo -arch=sm_53 --ptxas-options=-v --use_fast_math

CXXFLAGS:=$(OPT) -I. $(EXT_CXXFLAGS)
LDFLAGS:=-lm -lcudart $(EXT_LDFLAGS)

NVCFLAGS:=$(CXXFLAGS) $(NVOPT) --expt-relaxed-constexpr
NVLDFLAGS:=$(LDFLAGS) -lgomp

SRCS:= utils.c
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o) $(VERSION:%=$(BUILD_DIR)/%.o)
EXE=$(VERSION:.cu=.exe)

$(EXE):	$(OBJS)
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) $(OBJS) -o $@ $(NVLDFLAGS)

$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

all: $(EXE)

TEST ?= test.cu

test: $(BUILD_DIR)/tests/$(TEST).o
	$(MKDIR_P) $(dir $(BUILD_DIR)/tests/$(TEST))
	$(NVCC) $(NVCFLAGS) $< -o $(BUILD_DIR)/tests/$(TEST:.cu=.exe) $(NVLDFLAGS)
	$(BUILD_DIR)/tests/$(TEST:.cu=.exe)

$(BUILD_DIR)/tests/%.cu.o: tests/%.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) -c $< -o $@


.PHONY: run profile clean
run: $(EXE)
	./$(EXE)

profile: $(EXE)
	sudo $(CUDA_HOME)/bin/nvprof ./$(EXE)

clean:
	-rm -fr $(BUILD_DIR) *.exe *.out *~

MKDIR_P ?= mkdir -p
