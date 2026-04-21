# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -std=c++11 -I. -Isrc/hsieh -Isrc/reservation -Isrc/wei -Isrc/lin

# Target executable
TARGET = bin/main

# Source files
SRC = src/main.cpp src/hsieh/hsieh.cpp src/reservation/reservation.cpp src/wei/wei.cpp src/lin/lin.cpp

# Object files
OBJ = $(SRC:.cpp=.o)

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJ)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

# Compile each source file into an object file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJ) $(TARGET)