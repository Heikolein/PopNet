Compiler = nvcc
FLAGS = -O3

INC_DIRS = -Iinclude

TARGET = main
SRC_DIR = src
OBJ_DIR = build
CUDA_SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CUDA_SRCS))

$(TARGET): $(OBJS)
    $(Compiler) $(FLAGS) -o $@ $^ $(INC_DIRS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
    mkdir -p $(OBJ_DIR)
    $(Compiler) $(FLAGS) $(INC_DIRS) -c $< -o $@

clean:
    rm -f $(TARGET) $(OBJS)
