CC = g++
FLAGS = -Wall -g -I/usr/include/opencv -O0
MAIN = lab4exe
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV) -pthread
INCLUDES = *.hpp
OBJS = lab4.o
 
$(MAIN):$(OBJS)
	$(CC) $(FLAGS) -o $(MAIN) $(OBJS) $(LIBS)
 
%.o: %.cpp $(INCLUDES)
	$(CC) $(FLAGS) -c -o $@ $<
 
clean:
	rm -f $(OBJS) $(MAIN)
