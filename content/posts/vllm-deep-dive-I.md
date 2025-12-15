+++
Tags = [ "vLLM", "Python"]
Categories = ["Computer Science"]
date = '2025-12-13T16:20:32-08:00'
draft = false
title = 'vLLM Deep Dive I: Intro and Environment Setup'
[cover]
image = "/images/common/vllm-logo.png"
+++

## Introduction

It’s been a while since my last blog series on GGML. GGML is an excellent framework for running LLMs on-device — laptops, phones, and other resource-constrained environments — thanks to its pure C/C++ implementation and zero third-party dependencies. That makes it easy to use when performance and portability matter. That said, its design isn’t meant for industry-scale serving, where you need to manage thousands of high-end GPUs reliably.

Recently, I had the chance to dig into **vLLM**, probably the most popular inference framework for large-scale LLM serving. At a high level, vLLM’s architecture is quite clear and intuitive, but once you dive into the source code, the implementation details can feel vague and under-documented. There are plenty of guides on how to *use* vLLM — but not many that explain how it *works* under the hood.

After spending some time tinkering with the codebase, I think it makes sense to start a new blog series that does exactly that: walk through the internals of vLLM from a source-code perspective. This isn’t another tutorial about running `pip install vllm` or launching a server — it’s about understanding the design choices and core mechanisms that make vLLM tick.

A quick note on versions: vLLM is evolving fast, and structures and APIs can change significantly between releases. To keep things grounded, this series is based on [**v0.11.0**](https://github.com/vllm-project/vllm/tree/v0.11.0). At that version there are two engine implementations — **v0** and **v1**. They share some common components, but v1 is under active development while v0 is being deprecated. In this series, I’ll focus exclusively on the v1 engine.

## Environment Setup

Installing vLLM itself is fairly straightforward. The project provides precompiled wheels (vLLM includes both C++ and CUDA components), so in most cases you can just follow the official documentation and install everything with a single command:

```bash
pip install vllm==0.11.0
```

This works well if you just want to *use* vLLM. However, for this blog series, we’ll be reading and modifying the source code frequently—for example, dropping in `pdb.set_trace()` to understand control flow. Installing vLLM directly into `site-packages` isn’t very convenient for that.

Building vLLM entirely from source is possible, but it’s slow and usually unnecessary. Fortunately, vLLM provides a **Python-only install path** that reuses precompiled binaries and skips the full build process.

### Python-only editable install

For every commit on vLLM’s `main` branch, there is a corresponding precompiled wheel hosted on the official vLLM site. The idea here is simple: instead of compiling C++/CUDA code locally, we download the matching wheel, extract the compiled binaries, and wire them up to a local editable checkout of the Python code.

You can set this up as follows:

```bash
git clone git@github.com:vllm-project/vllm.git
cd vllm
git checkout v0.11.0

# Find the commit hash used by the v0.11.0 release
export VLLM_COMMIT=  # paste the output of: git merge-base v0.11.0 main

export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

pip install -e .
```

With this setup, any changes you make to the vLLM source code will take effect immediately on the next run—no rebuild required. This makes debugging and code exploration much more pleasant.

### A quick sanity check

Let’s try a simple example from vLLM’s quickstart documentation:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

Now try adding a breakpoint somewhere in vLLM's source code and run this script again.

…Oops. The process crashes.

This happens because vLLM launches multiple processes (at least three, which we’ll dig into in later posts). Naively inserting `pdb.set_trace()` tends to break the multiprocessing execution model.

The good news is that vLLM provides a simple escape hatch for debugging: you can disable multiprocessing entirely and force everything to run sequentially. Just set one extra environment variable:

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
```

Now rerun the script, and this time it should work as expected—even with breakpoints enabled. In addition, I’d recommend passing one extra argument when initializing the `LLM` class: `enforce_eager=True`. This disables `torch.compile` and CUDA graph capture. Without this, some breakpoints may only take effect once (which we’ll cover later in the series).

Now we have a clean and reliable starting point for inspecting vLLM’s internal behavior. In the next post, we’ll begin tracing what actually happens when `LLM.generate()` is called.

Have fun poking around vLLM’s source code, and see you in the next post :)
