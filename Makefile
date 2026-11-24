CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Iinclude
NVCC     ?= nvcc
ARCH     ?= sm_80
NVFLAGS  ?= -O2 -std=c++17 -arch=$(ARCH) -Iinclude

COMMON := src/main.cpp src/config.cpp src/output.cpp src/cpu_solver.cpp src/hybrid_solver.cpp

.PHONY: cpu gpu test-cpu test-hybrid check gui clean
cpu: wavesim
wavesim: $(COMMON) src/gpu_stub.cpp $(wildcard include/*.hpp)
	$(CXX) $(CXXFLAGS) -o $@ $(COMMON) src/gpu_stub.cpp

gpu: $(COMMON) src/gpu_solver.cu src/gpu_slab.cu $(wildcard include/*.hpp)
	$(NVCC) $(NVFLAGS) -o wavesim $(COMMON) src/gpu_solver.cu src/gpu_slab.cu

# Krish: CPU sanity checks (no config files needed)
test-cpu: test_cpu
	./test_cpu
test_cpu: tools/test_cpu.cpp src/cpu_solver.cpp $(wildcard include/*.hpp)
	$(CXX) $(CXXFLAGS) -o $@ tools/test_cpu.cpp src/cpu_solver.cpp

# Krish: hybrid split logic (two CPU slabs must equal CpuSolver exactly)
test-hybrid: test_hybrid
	./test_hybrid
test_hybrid: tools/test_hybrid.cpp src/hybrid_solver.cpp src/cpu_solver.cpp src/gpu_stub.cpp $(wildcard include/*.hpp)
	$(CXX) $(CXXFLAGS) -o $@ tools/test_hybrid.cpp src/hybrid_solver.cpp src/cpu_solver.cpp src/gpu_stub.cpp

# Krish: full CPU baseline check -- unit tests, then every config end to end
check: wavesim test_cpu test_hybrid
	./test_cpu --all-refs
	./test_hybrid
	tools/check_cpu.sh

# Krish: shared library for the live browser view (tools/live_gui.py)
gui: libwavesim_live.so
libwavesim_live.so: src/live_api.cpp src/cpu_solver.cpp $(wildcard include/*.hpp)
	$(CXX) $(CXXFLAGS) -fPIC -shared -o $@ src/live_api.cpp src/cpu_solver.cpp

clean:
	rm -f wavesim test_cpu test_hybrid libwavesim_live.so
