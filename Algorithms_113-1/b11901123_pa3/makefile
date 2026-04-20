# CC and CFLAGS are variables
CC = g++
CFLAGS = -c -std=c++11
AR = ar
ARFLAGS = rcv
# -c option asks g++ to compile the source files, but do not link.
# -g option is for debugging version
# -O2 option is for optimized version
DBGFLAGS = -g -D_DEBUG_ON_ -std=c++11
OPTFLAGS = -O2 -std=c++11

all	: bin/cb
	@echo -n ""

# optimized version
bin/cb	: main_opt.o
			$(CC) $(OPTFLAGS) main_opt.o -o bin/cb
main_opt.o 	   	: src/main.cpp 
			$(CC) $(CFLAGS) $< -o $@

# clean all the .o and executable files
clean:
		rm -rf *.o lib/*.a lib/*.o bin/*