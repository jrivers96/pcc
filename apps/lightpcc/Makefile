sources = LightPCC.cpp PearsonRMKL.cpp PearsonR.cpp SpearmanR.cpp KendallTau.cpp Distance.cpp MIAdaptive.cpp
src_dir = .
objs_dir = objs
objs = $(patsubst %.cpp, $(objs_dir)/%.cpp.o, $(sources))
objs_mpi = $(patsubst %.cpp, $(objs_dir)/%.mpi.o, $(sources))

#Maximum number of vectors, i.e. 2^25
macros = -DPR_MAX_MAT_DIM=33554432

#Use hand-tuned assemblies for single-precision
#macros += -DWITH_PHI_ASSEMBLY_FLOAT

#Use hand-tuned assemblies for double precision. Auto-vecto seems faster in practice!!!
#macros += -DWITH_PHI_ASSEMBLY_DOUBLE

#use ZLIB
#macros += -DCOMPRESSED_INPUT
#zlibs = -lz

#verbose
macros += -DVERBOSE

#library header files
LIB_HEADER_INC= -I ../../include

#Intel MKL (Only used for pearson's r)
MKL_ROOT=/opt/intel/mkl/
MKL_INC= -I $(MKL_ROOT)/include

#switch for Intel compiler vector report
#VEC_REPORT=-qopt-report=5 -qopt-report-phase=vec -qopt-report-file=stdout

#switch for the "-fast" compiling optation. May fail compilation when enabled for Intel Xeon Phis
#FAST= -fast

#compiler and options for non-MPI-based version
CXX = icpc
CXXFLAGS= -DHAVE_INTEL_ICC -O3 $(FAST) $(MKL_INC) -mkl -funroll-loops -Wall -I include $(macros) -qopenmp $(VEC_REPORT) $(LIB_HEADER_INC)
LIBS = $(MKL_LIB) -lm -loffload	$(zlibs)

#compiler and options for MPI-based version
MPICXX = mpiicpc
MPICXXFLAGS= -DHAVE_INTEL_ICC -O3 $(FAST) $(MKL_INC) -mkl -msse3 -funroll-loops -Wall -I include $(macros) -qopenmp $(VEC_REPORT) $(LIB_HEADER_INC)
MPILIBS= -lm -loffload $(zlibs)

#LightPCC
ppc_exec = PCC
mpi_ppc_exec = mpiPCC

all: dir shared distributed

dir:
	mkdir -p $(objs_dir)

shared: dir $(objs)
	$(CXX) $(CXXFLAGS) -o $(ppc_exec) $(objs) $(LIBS)
	strip $(ppc_exec)
	
distributed: dir $(objs_mpi)
	$(MPICXX) $(MPICXXFLAGS) -o $(mpi_ppc_exec) $(objs_mpi) $(MPILIBS)
	strip $(mpi_ppc_exec)

clean:
	-rm -rf $(objs_dir) $(ppc_exec) $(mpi_ppc_exec)

$(objs_dir)/%.cpp.o: $(src_dir)/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(objs_dir)/%.mpi.o: $(src_dir)/%.cpp
	$(MPICXX) $(MPICXXFLAGS) -DWITH_MPI -o $@ -c $<

