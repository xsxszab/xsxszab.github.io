+++
Tags = [ "Concurrent Programming", "C++" ]
Categories = ["Computer Science"]
date = '2023-12-15T16:35:38-08:00'
draft = false
title = 'Lock Free Cuckoo Filter'
ShowToc = true
[cover]
image = "/images/lock-free-cuckoo-filter/move.png"
+++

## Contributors

**Yifei Wang** ([xsxszab](https://github.com/xsxszab))  
**Yuchen Wang** ([yw7](https://github.com/darKred9))

## Summary

We implemented an efficient lock-free cuckoo filter in C++. Experiments demonstrate that our filter significantly outperforms its coarse-grained lock counterpart while reaching throughput comparable to the fine-grained lock version on both GHC and PSC platforms.

## 1. Background

### 1.1 Cuckoo Hash Table and Cuckoo Filter

#### Cuckoo Hashing

Cuckoo hashing is a type of open-addressing hash algorithm with $\mathcal{O}(1)$ worst-case lookup time. It uses two hash functions, $hash_1$ and $hash_2$, to identify two potential locations for a key. If any of these two locations is empty, the key will be inserted there. Otherwise, Cuckoo hashing kicks out one item, places the new key, and moves the displaced key to its alternative location. This replacement process continues until an empty entry is found.

<!-- ![](/assets/images/hash.png) -->
{{< figure
  src="/images/lock-free-cuckoo-filter/hash.png"
>}}

#### Cuckoo Filter

Proposed in [1], the cuckoo filter is built on the 4-way associative cuckoo hash table. Instead of storing the key, it stores a fingerprint (typically shorter than the key) to reduce memory usage. To locate a key's alternative position, partial key cuckoo hashing is used:

$$
h_1(x) = hash(x), \\
h_2(x) = h_1(x) \oplus hash(fprint(x))
$$

<!-- ![](/assets/images/table.png) -->
{{< figure
  src="/images/lock-free-cuckoo-filter/table.png"
>}}

### 1.2 Coarse-grained Lock Cuckoo Filter

A global mutex locks the entire table, serializing all operations. This version serves as a baseline in our experiments.

### 1.3 Fine-grained Lock Cuckoo Filter

Instead of a global lock, this version assigns multiple locks to different sections of the table. To improve concurrency, read operations use separate locks from write operations.

## 2. Lock-free Cuckoo Filter

### 2.1 Data Structures

#### 4-way Associative Cuckoo Hash Table

Our table stores fingerprints in dynamically allocated memory, referenced by 64-bit augmented pointers that allow atomic operations.

<!-- ![](/assets/images/pointer.png) -->
{{< figure
  src="/images/lock-free-cuckoo-filter/pointer.png"
>}}

#### Table Pointer Structure

The pointer structure includes:

- **48-bit real pointer**: Stores the fingerprint address.
- **16-bit counter**: Tracks relocation occurrences.
- **1-bit relocation flag**: Indicates if an entry is being relocated.

### 2.2 Find, Insert, and Remove Operations

#### Find Operation

The find operation may miss existing fingerprints due to concurrent relocation. To prevent false misses, we track entry counters and perform a two-round search.

<!-- ![](/assets/images/move.png) -->
{{< figure
  src="/images/lock-free-cuckoo-filter/move.png"
>}}

#### Insert Operation

1. Locate an empty entry.
2. If found, use CAS (Compare-and-Swap) to insert the fingerprint.
3. If not, invoke the **relocate** operation.

If relocation fails, the insert operation fails due to insufficient capacity.

#### Remove Operation

The remove operation:

1. Uses the find operation to locate the fingerprint.
2. Uses CAS to delete it.
3. Adds the deleted pointer to the retired list for safe memory management.

### 2.3 Relocate Operation

Relocation finds a replacement path and shifts fingerprints until an empty entry is created. If no empty entry is found within a predefined threshold, the operation fails.

<!-- ![](/assets/images/relocate.png) -->
{{< figure
  src="/images/lock-free-cuckoo-filter/relocate.png"
>}}

### 2.4 Hazard Pointer Based Memory Management

To ensure safe memory reclamation:

- **Hazard Pointers** track active memory locations.
- **Retired Pointers** store removed fingerprints awaiting deletion.

Once a thread accumulates enough retired pointers, it deletes those not found in any hazard pointer list.

## 3. Experimental Results

### 3.1 Performance Comparison on GHC Machines

Experiments show that the lock-free cuckoo filter outperforms the coarse-grained lock version and performs comparably to the fine-grained lock version.

<!-- ![](/assets/images/ghc_compare.png) -->
{{< figure
  src="/images/lock-free-cuckoo-filter/ghc_compare.png"
>}}

### 3.2 Performance Comparison on PSC Machines

On PSC machines, both fine-grained and lock-free versions perform similarly. Performance drops when the thread count exceeds 32 due to contention.

<!-- ![](/assets/images/psc_compare.png) -->
{{< figure
  src="/images/lock-free-cuckoo-filter/psc_compare.png"
>}}

### 3.3 Impact of Hash Table Associativity

Higher associativity reduces performance due to increased iteration costs in operations.

### 3.4 Impact of Load Factor

Performance declines as load factor increases because inserts require more searches and relocations.

<!-- ![](/assets/images/ghc_load_factor.png) -->
{{< figure
  src="/images/lock-free-cuckoo-filter/ghc_load_factor.png"
>}}

## 4. Future Work

- **Lifting Relocation Restrictions**: A BFS-based approach may reduce the relocation path length.
- **SIMD Optimization**: Using SIMD instructions to accelerate operations.
- **Fairness in Fine-grained Locking**: Addressing reader-writer priority issues.

## 5. Conclusion & Acknowledgement

We implemented a lock-free cuckoo filter with superior performance in highly concurrent environments. Due to time constraints, some optimizations were not completed, but overall, the project was successful.

We acknowledge the inspiration from [1] and [2], and the usage of OpenSSL and Cycletimer from the 15418 course resources.

## 6. Work Distribution

| Task | Yifei Wang | Yuchen Wang |
|-------------------------------|:-----------:|:------------:|
| Implement coarse-grained lock version | ✔ |  |
| Implement fine-grained lock version |  | ✔ |
| Implement lock-free version | ✔ | ✔ |
| Debugging and Profiling | ✔ | ✔ |
| Project Proposal | ✔ |  |
| Milestone Report | ✔ |  |
| Final Report | ✔ | ✔ |
| Project Poster | ✔ | ✔ |

## References

[1] Bin Fan, Dave G. Andersen, Michael Kaminsky, and Michael D. Mitzenmacher. "Cuckoo Filter: Practically Better Than Bloom." CoNEXT ‘14.

[2] N. Nguyen and P. Tsigas. "Lock-Free Cuckoo Hashing." ICDCS 2014.

[3] Bin Fan, David G. Andersen, and Michael Kaminsky. "MemC3: Compact and concurrent MemCache with dumber caching and smarter hashing." NSDI’13.

[4] M. M. Michael. "Hazard pointers: safe memory reclamation for lock-free objects." IEEE TPDS, 2004.

## Appendix

- Project repository: [GitHub](https://github.com/xsxszab/lock_free_cuckoo_filter)
