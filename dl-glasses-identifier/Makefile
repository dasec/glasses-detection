CXX=g++
# Linux
IDIR=
CXXFLAGS=-O2 -Wall -fmessage-length=0 -std=c++11 -DCPU_ONLY -DUSE_OPENCV -DNDEBUG
LINKFLAGS=-s
LDLIBS=`pkg-config --libs opencv` -lglog -lprotobuf -lboost_system -lboost_filesystem -lboost_thread -lcblas -pthread -lcaffe
#-lhdf5_serial_hl -lhdf5_serial 
SRCS=src/Main.cpp
OBJS=$(subst src/,build/,$(subst .cpp,.o,$(SRCS)))

all: glasses-identifier

glasses-identifier: $(OBJS)
	@mkdir -p build/
	$(CXX) $(LINKFLAGS) -o build/glasses-identifier $(OBJS) $(LDLIBS)

build/%.o: src/%.cpp 
	@mkdir -p build/
	$(CXX) $(IDIR) -c $(CXXFLAGS) $< -o $@

clean:
	rm $(OBJS)

dist-clean: clean
	rm glasses-identifier

