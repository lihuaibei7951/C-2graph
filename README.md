# C-2graph
C^2graph: A Compression-Collaboration Algorithm for CPU-GPU Hybrid Weighted Graph Traversals

## 1. Introduction
This is the open-source implementation for C^2graph.

With the continuous growth of graph data, the demand for graph traversal queries in multi-tenant environments has been increasing. However, the necessity of traversing numerous edges and their weights makes traditional graph algorithms highly inefficient in terms of both time and space complexity. Existing distributed or parallel solutions alleviate these issues but often require substantial hardware investment and suffer from high communication delays. Recent studies have explored the use of modern GPUs and CPUs on a single device, yet these methods still face limitations in memory scalability and underutilization of compute resources.

This paper addresses the above challenges by investigating the redundancy in graph traversal behaviors and proposes **C^2graph-P** that prunes not only the graph topology but also edge weights, significantly reducing memory usage. To ensure correctness, the paper compensates for the information loss caused by pruned edges and provides a theoretical guarantee.

To further enhance the performance of graph traversal queries, the paper introduces **C^2graph-PM**. This mechanism enables multiple traversal tasks to adaptively select between GPUs and CPUs based on peak computational demands. By optimizing CPU-GPU collaboration costs, the proposed framework fully leverages the computational power of both, thus improving query throughput.

### **Key Contributions**
1. **Efficient Compression Algorithm**:
   - Reduces memory overhead by pruning redundant graph topology and edge weights.
   - Ensures correctness through effective compensation for pruned edges.

2. **Dynamic Workload Allocation**:
   - Adapts to dynamic query workloads by selecting appropriate accelerators (GPU or CPU).
   - Balances task allocation to maximize the utilization of heterogeneous computing devices.

3. **Optimized CPU-GPU Collaboration**:
   - Minimizes communication overhead between CPUs and GPUs.
   - Maximizes hardware resource utilization to enhance graph traversal performance.


## 2. Quick Start

Before running it, some softwares must be installed, which is beyond the scope of this document. 
### 2.1 Requirements
nvcc-version = 10.1
gcc/g++ 7.5
### 2.2 Compilation
```bash
git clone https://github.com/lihuaibei7951/C-2graph.git --recursive
cd CompressGraph
mkdir -p build
cd build
cmake ..
make -j
```
### 2.3 Graph Input Format
example
### 2.4 C-2graph-pri
example
### 2.5 C-2graph-P
example
### 2.6 C-2graph-PM
example
## 3. Contact  
If you encounter any problem with LRCNN, please feel free to contact lihuaibei7951@stu.ouc.edu.cn.

