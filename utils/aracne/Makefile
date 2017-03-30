sources = convert2aracne.cpp
src_dir =.
objs_dir = objs
objs = $(patsubst %.cpp, $(objs_dir)/%.cpp.o, $(sources))

CXX=g++
CXXFLAGS = -O3 -fopenmp -funroll-loops -ffast-math -Wall -I .
LIBS = -lm -lz

#executables
EXEC = convert2aracne

all: dir $(objs)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(objs) $(LIBS)
	strip $(EXEC)

dir:
	mkdir -p $(objs_dir)

clean:
	-rm -rf $(objs_dir) $(EXEC)

$(objs_dir)/%.cpp.o: $(src_dir)/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

