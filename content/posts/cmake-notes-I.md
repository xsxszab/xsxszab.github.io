+++
Tags = [ "CMake", "C++" ]
Categories = ["Computer Science"]
date = '2025-03-01T10:49:01-08:00'
draft = false
title = 'CMake Notes I: Different Scopes in Cmake'
ShowToc = true
[cover]
image = "/images/common/cmake-logo.png"

+++

# I Hate CMake

CMake is the most popular tool for managing C++ projects of any scale—and for good reason. It's powerful, flexible, and capable of handling almost any build requirement. If you think it's missing a feature, you best guess is it already exists—you just haven't found it yet. But to be honest, I've never liked CMake. Writing CMakeLists feels like mental torture, thanks to its many quirks and weird behaviors.

{{< figure
  src="/images/cmake-notes-I/meme.png"
>}}


To make my life easier, I've started this series as a collection of notes on CMake’s behavior—both for my own reference and for anyone else struggling with it. In this post, I'll dive into the different scopes in CMake and how they work.

---

# Scopes in CMake
First things first—forget everything you know about scoping rules in programming languages like C/C++, Python, or Java. Just erase it from your mind for now. CMake's scoping rules are nothing like them, and trying to apply traditional logic here will only lead to confusion.

## 1. Normal Variables (`set(VAR value)`)

### **Scope in Current CMakeLists**

- By default, variables defined with `set(VAR value)` can be accessed in current CMakeLists after the lien they are defined.
```cmake
# Empty output. In CMake, accessing an undefined variable won't trigger any error
message(STATUS "${TEST_VAR}")

set(TEST_VAR "a")

message(STATUS "${TEST_VAR}") # Output: a
```
### **Scope in** `add_subdirectory()`

- Variables from a parent directory are available in subdirectories.
- Subdirectories can override the variables, but these changes do **NOT** propagate back to the parent.

```cmake
# root CMakeLists.txt
set(TEST_VAR "a")
add_subdirectory(subdir)
message(STATUS "parent var: ${TEST_VAR}") # Output: a

# subdir CMakeLists.txt
message(STATUS "subdir var before: ${TEST_VAR}") # Output: a
set(TEST_VAR "b") # Overrides in subdir scope
message(STATUS "subdir var after: ${TEST_VAR}") # Output: b
```

### **Behavior in Functions**

- Functions **create a new scope** for variables.
- Variables set inside functions **do not affect the caller's scope** unless explicitly marked `PARENT_SCOPE`.

```cmake
function(test_func)
    set(TEST_VAR "b")
endfunction()

set(TEST_VAR "a")
test_func()
message(STATUS "TEST_VAR: ${TEST_VAR}") # Outputs: a
```

To modify the variable in parent scope, add `Parent_scope`:

```cmake
function(test_func)
    set(TEST_VAR "b" PARENT_SCOPE)
endfunction()
```

### **Behavior in Macros**

- Macros do **NOT** introduce new scope.
- Variables modified in a macro affect the caller's scope.

```cmake
macro(test_macro)
    set(TEST_VAR "b")
endmacro()

set(TEST_VAR "a")
test_macro()
message(STATUS "TEST_VAR: ${TEST_VAR}") # Outputs: b
```

## 2. Cache Variables (`set(VAR value CACHE)` or `option` )

### Scope and Lifetime

- Like normal variables, they can only be accessed after the line they are defined during the first run of CMake.
- Once set, their values are stored in CMake's build directory and can be accessed across any CMake runs unless explicitly deleted/overridden.
- Cache variables are defined in `{build_dir}/CMakeCache.txt`. Delete it (or the entire build directory) if you need a completely clean build.

```cmake
cmake_minimum_required(VERSION 3.14)
project("test-cmake" C CXX)

# Run this script for the first time: no output
# On subsequent runs, the output will be: a
message(STATUS "${TEST_VAR}")

set(TEST_VAR "a" CACHE STRING "a global variable")
```
- Modifying a cache variable  requires `FORCE`:


```cmake
function(my_func)
    set(MY_CACHE_VAR "New Value" CACHE STRING "Description" FORCE)
endfunction()
```

### Behavior in `add_subdirectory()`, Functions and Macros 

- Same as normal variables.

### Behavior of Variables defined by  `option()` 

- Variables defined using `option(VAR "Description" DEFAULT_VALUE)` behave almost the same as cache variables.
- The type of variables defined using `option` will always be Bool.
- If the variable is already defined in the cache, `option()` does **NOT** override it.

## 3. Environment Variables (`$ENV{VAR}`)

### Scope and Lifetime

- Environment variables are defined outside CMake but are accessible within CMake using `$ENV{VAR}`.
- Changes to environment variables in CMakeLists persist only for the current CMake process (**NOT** system-wide).

```cmake
set(ENV{ENV_VAR} "test string")
message(STATUS "ENV_VAR: $ENV{ENV_VAR}")
```

### Behavior in Subdirectories, Functions and Macros

- Same as normal variables. 

## 4. Variables in `ExternalProject_Add()`

- `ExternalProject_Add()` runs a completely separate CMake process, which implies that environment variables defined in parent CMakeLists won't affect current external project.
- Parent variables are not automatically passed to the external project, you must pass values explicitly using `-DVAR=value`.

```cmake
ExternalProject_Add(test_external_project
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/external_project_dir
    CMAKE_ARGS -DTEST_VAR=${TEST_VAR_IN_CURREENT_SCOPE}
)
```

## 5. Functions and Macros

* Here we are talking about the scope of functions and macros themselves, not the variables defiend inside them.
* Functions and macros, once defined, are globally accessable.
```cmake
# root CMAkeLists
cmake_minimum_required(VERSION 3.10)
project(FunctionScopeDemo)

add_subdirectory(subdir)
subdir_func() # output: hello world

# subdir CMakeLists
function(subdir_func)
    message(STATUS "hello world")
endfunction()

```
* Unlike variables, attempting to call an undefined function or macro will result in error.

## 6. Nested Functions
* Unlike languages with lexical scoping for nested functions, CMake defines inner functions globally—but only if the outer function runs. If the outer function never executes, the inner function doesn’t exist at all.

* The same rule applies to macros as well.

  ```cmake
  function(outer_function)
      function(inner_function)
          message(STATUS "inner function runs")
      endfunction()
  
      message(STATUS "outer function runs")
  endfunction()
  
  outer_function()
  inner_function()
  
  # Output:
  # outer function runs
  # inner function runs
  
  # If `outer_function` is commented out, running this script will result in error.
  ```

  
## Summary Table

| Variable Type  | Global?                         | Visible in `add_subdirectory()`? | Affects Parent Scope when defined in functions? | Available in External Projects? | Persistent Across Runs? |
| -------------- | ------------------------------- | -------------------------------- | ----------------------------------------------- | ------------------------------- | ----------------------- |
| Normal         | No                              | Yes                              | No (unless `PARENT_SCOPE` is used)              | No                              | No                      |
| Cache          | Yes                             | Yes                              | No (unless `FORCE` is used)                     | No                              | Yes                     |
| Environment    | Yes (for current CMake process) | Yes                              | No                                              | No                              | No                      |
| Function/Macro | Yes                             | Yes                              | Yes (note #6)                                   | No                              | No                      |
