CC = g++
FLAGS = -Wall -g -O0 -mfpu=neon -mcpu=Cortex-A8 -ftree-vectorize -std=c++11
MAIN = lab5exe
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV) -pthread
INCLUDES = *.hpp
OBJS = lab5.o
 
$(MAIN):$(OBJS)
	$(CC) $(FLAGS) -o $(MAIN) $(OBJS) $(LIBS)
 
%.o: %.cpp $(INCLUDES)
	$(CC) $(FLAGS) -c -o $@ $<
 
clean:
	rm -f $(OBJS) $(MAIN)
