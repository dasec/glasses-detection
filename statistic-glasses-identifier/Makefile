CXX=g++
# Linux
CXXFLAGS=-O3 -fopenmp -Wall -fmessage-length=0 -std=c++14 -DNDEBUG
LINKFLAGS=-s -L/usr/include/ -fopenmp
LDLIBS=-pthread -lboost_system -lboost_filesystem -lboost_regex `pkg-config --libs opencv` -lglog -lmatio

SRCS=src/Main.cpp
OBJS=$(subst src/,build/,$(subst .cpp,.o,$(SRCS)))

all: glasses-identifier

glasses-identifier: $(OBJS)
	@mkdir -p build/
	$(CXX) $(LINKFLAGS) -o build/glasses-identifier $(OBJS) $(LDLIBS)

build/%.o: src/%.cpp
	@mkdir -p build/
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm $(OBJS)

dist-clean: clean
	rm glasses-identifier
