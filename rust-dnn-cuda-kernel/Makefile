CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

SRC := src/cuda/basic.cu src/cuda/math.cu src/cuda/utils.cu src/cuda/nn.cu
OUT_DIR := target
OBJ := $(SRC:.cu=.o)
OBJ := $(OBJ:%=$(OUT_DIR)/%)

LIB := $(OUT_DIR)/libkernel.a
DLINK_OBJ := $(OUT_DIR)/dlink.o

all: $(LIB)

# CUDA 静的ライブラリ作成 + デバイスリンク対象
$(LIB): $(OBJ) $(DLINK_OBJ)
	ar rcs $@ $(OBJ) $(DLINK_OBJ)

# 各 .cu → .o
$(OUT_DIR)/%.o: %.cu
	mkdir -p $(dir $@)
	$(NVCC) -c -Xcompiler -fPIC -rdc=true $< -o $@

# デバイスリンク用オブジェクト
$(DLINK_OBJ): $(OBJ)
	$(NVCC) -dlink -Xcompiler -fPIC -o $@ $^

clean:
	rm -rf $(OUT_DIR)
