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

This section will guide you through the installation and usage of C-2graph. Before running the program, certain software dependencies must be installed, which are outlined below.
### 2.1 Requirements
nvcc-version >= 11.0(The CUB library was removed from the CUDA Toolkit starting from CUDA 11.0.)

gcc/g++ 7.5
### 2.2 Compilation
Once the prerequisites are met, follow these steps to compile the C-2graph project:
```bash
git clone https://github.com/lihuaibei7951/C-2graph.git --recursive
cd C-2graph
mkdir -p build
cd build
cmake ..
make -j
```
### 2.3 Graph Input Format&&get binary file
C-2graph expects the graph data in a specific text-based adjacency list format. 
Each line in the input file represents a vertex and its outgoing edges. For example:
```
src1   dst1:dst2
src2   dst3:dst4:dst5
```
To convert the graph into binary format, use the convert2binary tool, 
which will generate the necessary binary files for further processing.

Run the following command:
```
../bin/convert2binary input_file_path  dataset_path

```
out: vlist.bin  elist.bin wlist.bin(int) flist.bin(float)
### 2.4 C-2graph-pri
The C-2graph-pri utility is used to preprocess the graph and run the primary graph algorithms. 
It is essential for preparing the graph data before executing specific algorithms.

To run this step, use the following command:
```
../bin/purn-sssp dataset_path

```
This command will perform preprocessing on the graph data stored at dataset_path, 
preparing it for efficient execution of graph algorithms like Single-Source Shortest Path (SSSP).
### 2.5 C-2graph-P
The C-2graph-P utility provides two versions of the Single-Source Shortest Path (SSSP) algorithm:

sssp-base: The baseline implementation of SSSP.

sssp-purn: The optimized version of SSSP, incorporating preprocessing for faster execution.

To run and compare the results of both algorithms, use the following commands:
```
../bin/sssp-base dataset_path source
../bin/sssp-purn dataset_path source
```
### 2.6 C-2graph-PM
In this program, we need to pay attention to the migration of actual parameters during the switch.

sssp-purn-M: Optimized SSSP with multi-threading/GPU support.

To run C-2graph-PM, use the following commands:

```
../bin/sssp-purn-M dataset_path source migration
```
## 3. Contact  
If you encounter any problem with C-2graph, please feel free to contact lihuaibei7951@stu.ouc.edu.cn.

