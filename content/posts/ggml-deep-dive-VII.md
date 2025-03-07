+++
Tags = [ "GGML", "C++" ]
Categories = ["Computer Science"]
date = '2025-03-07T17:13:23-08:00'
draft = true # TODO: remember to unmark this once finished
title = 'GGML Deep Dive VII: Tensor Representaion and In-memory Layout'
[cover]
image = "/images/common/ggml.png"
+++

# Introduction
In prevous posts, we have encountered the concept "tensor" in GGML for many times. But we have only seen its simplest usage: cases with no quantization, no permutation (i.e., the tensor has continuous in-memory layout) and no tensor view. In more complex use caes, the tensor has much more complicated behaviors and sometimes even counterintuitive. In this blog post, I'll do a more in-depth of tensors in GGML.

# ggml_tensor Data Structure

## Overview

First, let's take a look at the `ggml_tensor` data structure defined in `include/ggml.h`. Here are some fields that you should pay attention to:
* `enum ggml_type type`: The tensor's data type (e.g., GGML_TYPE_F32, GGML_TYPE_Q4_0, etc.).
* `enum ggml_op op`: The operator that produces this tensor (e.g., GGML_OP_ADD, GGML_OP_MATMUL, etc.).
* `char name[GGML_MAX_NAME]`: The name of the tensor.
* `struct ggml_tensor *src[GGML_MAX_SRC]`: Pointers to the parent tensors of this tensor.
* `struct ggml_tensor *view_src`: Points to the underlying tensor if this tensor is a view of another tensor (explained later).
* `size_t view_offs`: view offset, used together with view_src .
* `int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)]`: Operator-specific parameters (e.g., dimension information for `ggml_permute`).
* `int64_t ne[GGML_MAX_DIMS]`: Tensor shape information (important, explained later).
* `size_t nb[GGML_MAX_DIMS]`: Strides in each dimension (important, explained later).

## How to Setup a Simple Testing Environment
It doesn't help by just staring at the source code. To truly understand the internal mechanisms we are going to explore in this blog post, you need to setup a debuggin environment to observe GGML's runtime behavior. Here's a minimal setup for inspecting `ggml_tensor`'s behavior:
1. Create a new folder `tensor-test`under `examples/`.
2. In `examples/tensor-test`,create a `CMakeLists.txt`will the following content:
```cmake
add_executable(tensor-test main.cpp)
target_compile_options(tensor-test PRIVATE -g)
target_compile_options(tensor-test PRIVATE -O0)
target_link_libraries(tensor-test PRIVATE ggml)
```

3. Edit `examples/CMakeLists.txt`, add one line `add_subdirectory(tensor-test)`.
4. In `examples/tensor-test`, create `main.cpp` :
```c++
#include "ggml.h"

#include <iostream>

void print_dim(ggml_tensor* tensor) {
    for(int i=0; i<4; i++) {
        std::cout << tensor->ne[i] << " ";
    }
    std::cout << std::endl;
}

int main () {
    struct ggml_init_params params {
        /*.mem_size   =*/ 32 * 256 * 256,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    ggml_context * ctx = ggml_init(params);
    
    //write any tensor definition/operator here as you want 
    ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 4, 10, 20);
    ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 2, 100, 200);
    ggml_tensor* result = ggml_mul_mat(ctx, a, b);
    
    ggml_free(ctx);
    return 0;
}
```

## Understanding ggml_tensor.ne (Logical Tensor Layout)

## Understanding ggml_tensor.nb (In-memory Tensor Layout)

## Tensor Views

# Wrapping Up
