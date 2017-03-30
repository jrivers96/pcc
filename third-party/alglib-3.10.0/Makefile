sources = alglib_pr.cpp alglibinternal.cpp ap.cpp diffequations.cpp integration.cpp linalg.cpp solvers.cpp statistics.cpp alglibmisc.cpp dataanalysis.cpp fasttransforms.cpp interpolation.cpp optimization.cpp specialfunctions.cpp

src_dir =.
objs_dir = objs
objs = $(patsubst %.cpp, $(objs_dir)/%.cpp.o, $(sources))

#select compiler to use
COMPILER=intel
ifneq ($(COMPILER), intel)
CXX=g++
CXXFLAGS = -O3 -fopenmp -funroll-loops -ffast-math -Wall -I include $(macros) -I . -I ../../include
LIBS = -lm $(zlibs)

MPICXX=mpic++
MPICXXFLAGS = -O3 -fopenmp -funroll-loops -ffast-math -Wall -I include $(macros) -I . -I ../../include
MPILIBS = -lm $(zlibs)
else
#VEC_REPORT=-qopt-report=5 -qopt-report-phase=vec -qopt-report-file=stdout -DAE_CPU=AE_INTEL
CXX = icpc
CXXFLAGS= -O3 -funroll-loops -Wall -I . -I include $(macros) $(VEC_REPORT) -qopenmp -I ../../include
LIBS= -lm $(zlibs)

MPICXX = mpiicpc
MPICXXFLAGS= -O3 -fast  -funroll-loops -Wall -I . -I include $(macros) -qopenmp $(VEC_REPORT) -I ../../include
MPILIBS= -lm $(zlibs)
endif

#executables
EXEC = alglib_pr
all: dir shared

dir:
	mkdir -p $(objs_dir)

shared: dir $(objs)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(objs) $(LIBS)
	strip $(EXEC)
	
clean:
	-rm -rf $(objs_dir) $(EXEC)

$(objs_dir)/%.cpp.o: $(src_dir)/%.cpp
	$(CXX) $(CXXFLAGS) -DWITH_PHI -o $@ -c $<

