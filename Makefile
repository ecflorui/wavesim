CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Iinclude
NVCC     ?= nvcc
ARCH     ?= sm_80
NVFLAGS  ?= -O2 -std=c++17 -arch=$(ARCH) -Iinclude

COMMON := src/main.cpp src/config.cpp src/output.cpp src/cpu_solver.cpp

.PHONY: cpu gpu test-cpu clean
cpu: wavesim
wavesim: $(COMMON) src/gpu_stub.cpp $(wildcard include/*.hpp)
	$(CXX) $(CXXFLAGS) -o $@ $(COMMON) src/gpu_stub.cpp

gpu: $(COMMON) src/gpu_solver.cu $(wildcard include/*.hpp)
	$(NVCC) $(NVFLAGS) -o wavesim $(COMMON) src/gpu_solver.cu

# Krish: CPU sanity checks (no config files needed)
test-cpu: test_cpu
	./test_cpu
test_cpu: tools/test_cpu.cpp src/cpu_solver.cpp $(wildcard include/*.hpp)
	$(CXX) $(CXXFLAGS) -o $@ tools/test_cpu.cpp src/cpu_solver.cpp

clean:
	rm -f wavesim test_cpu
