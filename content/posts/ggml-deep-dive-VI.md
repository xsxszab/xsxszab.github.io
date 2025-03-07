+++
Tags = [ "GGML", "C++" ]
Categories = ["Computer Science"]
date = '2025-03-06T17:13:23-08:00'
draft = false
title = 'GGML Deep Dive VI: GGUF File Parsing'
[cover]
image = "/images/common/ggml.png"
+++

# Introduction
Initially, I planned to go over the `mnist` example in this post, but after looking into it, I realized that most of its key points had already been covered in previous posts : ) So instead, I’ll focus on the only part that hasn’t been discussed yet: the GGUF file format and how GGML parses it.

> **Note:** This blog series is based on **commit** [**475e012**](https://github.com/ggml-org/ggml/tree/475e01227333a3a29ed0859b477beabcc2de7b5e).

# GGUF File Structure

## Overview
GGML uses a binary file format called GGUF to store models, including metadata and tensor weights. Below is an overview of its structure from GGML's official 
[documentation](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md):

{{< figure
  src="/images/ggml-deep-dive-VI/gguf.png"
>}}

However, this diagram omits many important details. To truly understand the file structure, the best approach is to inspect it using a binary file reader. A good starting point is the `mnist-fc-f32.gguf` file. You can generate this file by following the instructions in the `mnist` example’s README, or you can download the one I've generated (~1.5MB):

[mnist-fc-f32.gguf](/files/mnist-fc-f32.gguf)

## How to Manually Inspect a GGUF File in VSCode
In the first blog of this series, I mentioned a VSCode plugin called [Hex Editor](https://marketplace.visualstudio.com/items?itemName=ms-vscode.hexeditor), which is perfect for reading GGUF files. After installing the plugin, right-click on the GGUF file in the left panel, select **"Open with..."**, and choose **"Hex Editor"**. Your window should look like this:

{{< figure
  src="/images/ggml-deep-dive-VI/code1.png"
>}}

## Reading a GGUF File
The first few sections of the GGUF file (magic number, version, tensor count, etc.) are straightforward to read since they have fixed lengths. While this GGUF file uses little-endian format, which is the only support format in old GGML versions, keep in mind that the latest GGML version also supports big-endian files.

Things get more interesting when we reach the key-value pair section. Before diving in, let’s take a quick look at GGML’s GGUF parser implementation in `src/gguf.cpp -> gguf_init_from_file_impl`. You don’t need to read through it entirely—manually inspecting the file is an easier way to understand the format—but we’ll refer to the code when needed.

Now, let’s examine the first (and only) key-value (KV) pair in this file. In GGUF, keys are always strings, and strings are stored as an 8-byte length value followed by the actual string (without a null terminator), as shown below:

{{< figure
  src="/images/ggml-deep-dive-VI/code2.png"
>}}

Following the key is a 4-byte type enum indicating the value's type (string, boolean, integer, etc.). Here, the value is `8`, which corresponds to `GGUF_TYPE_STRING` (defined in `include/gguf.h`), meaning the value is also a string. In this case, it is `"mnist-fc"`.

{{< figure
  src="/images/ggml-deep-dive-VI/code3.png"
>}}

Next, let’s look at the first tensor info section for `"fc1.weight"`—probably the most confusing part. Be sure to refer to the GGUF structure diagram above.


```
| (18 bytes) string "fc1.weight" | (4 bytes) number of dimensions = 2 | (8 bytes * 2) two int64 dimension values | (4 bytes) data type GGML_TYPE_F32 | (8 bytes) offset |
```
Parsing the first part of the tensor info section—name, number of dimensions, and dimension values—is straightforward. However, right after that, you'll encounter a series of zeros. These correspond to the data type and offset fields, both of which are zero because:
- `GGML_TYPE_F32 = 0` (defined in `include/ggml.h`).
- This is the first tensor in the file, so it has a zero offset.

All remaining tensor info sections follow the same structure, and the "rest of the file" section contains actual tensor weights.

# Wrapping Up
This blog ended up being much shorter than expected, as the only noteworthy part of `mnist` we haven't talked about is GGUF parsing. In the next post, I won’t be covering another example. Instead, I’ll revisit two key concepts in GGML: computational graph and tensor. Previously, I only briefly discussed their usage and corresponding data structure, but it has more complex behaviors in certain cases—especially when quantization is applied or when a tensor is permuted. The next post will take a deeper look at these behaviors and how they could impact inference performance. See you then!