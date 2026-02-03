# Compiler and flags
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

NVCC = nvcc

CFLAGS = \
	-O3 -arch=sm_75 \
    -gencode arch=compute_75,code=sm_75 \
    -Xcompiler -Wall,-march=native

LDFLAGS = -lcublas -lcusolver -lcudart

TARGET = $(BIN_DIR)/matrix_ops_cuda

SOURCES = $(SRC_DIR)/matrix_ops_cuda.cu
OBJECTS = $(OBJ_DIR)/matrix_ops_cuda.o

all: $(TARGET)

$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(NVCC) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
