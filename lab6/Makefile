CC = g++
FLAGS = -Wall -g -O0 -mfpu=neon -mcpu=Cortex-A8 -ftree-vectorize -std=c++11 -I/usr/local/include/perf -I/usr/local/lib/perf/include  -L/usr/local/lib -lperf
MAIN = lab6exe
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV) -pthread
INCLUDES = 
OBJS = lab6.o
 
$(MAIN):$(OBJS)
	$(CC) $(FLAGS) -g -o $(MAIN) $(OBJS) $(LIBS)
 
%.o: %.cpp $(INCLUDES)
	$(CC) $(FLAGS) -c -o $@ $<
 
clean:
	rm -f $(OBJS) $(MAIN)
