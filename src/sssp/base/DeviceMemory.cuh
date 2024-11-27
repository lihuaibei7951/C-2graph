#include "Graph.h"

class DeviceMemory {
  public:
    Vertex *csr_v;
    Vertex *csr_e;
    ValueType *csr_w; //i.e., reserve
    
    ValueType *distance; //i.e., reserve

	bool      *isactive;
	Vertex *active_vert;
    Vertex * active_vert_num;

	
     int *iteration_num;
	int *iteration_act_num;


    
    //size_t pitch; // 存储实际行大小的变量

    DeviceMemory(int _v_num, int _e_num)
        : vert_num(_v_num), edge_num(_e_num) {

        csr_v = NULL;
        csr_e = NULL;
        csr_w = NULL;
        distance = NULL;
        
        active_vert = NULL;
        active_vert_num = NULL;
        isactive = NULL;
        iteration_num = NULL;
	   iteration_act_num =NULL;
        

        CudaMallocData();
        cout << "INIT--class DeviceMemory is constructed" << endl;
    }

    ~DeviceMemory() {
        CudaFreeData();
        cout << "CLEAR--class DeviceMemory is destroyed" << endl;
    }

    // Copy graph data from host to device.
    void CudaMemcpyGraph(Graph &graph) {

        Vertex *resultArray = vec_to_arr(graph.csr_e);
        CUDA_ERROR(cudaMemcpy(csr_e, resultArray, sizeof(Vertex) * edge_num,
                              cudaMemcpyHostToDevice));
        delete[] resultArray;

        ValueType *result = vec_to_arr(graph.csr_w);
        CUDA_ERROR(cudaMemcpy(csr_w, result, sizeof(ValueType) * edge_num,
                              cudaMemcpyHostToDevice));
        delete[] result;

        resultArray = vec_to_arr(graph.csr_v);
        CUDA_ERROR(cudaMemcpy(csr_v, resultArray, sizeof(Vertex) * (vert_num + 1),
                              cudaMemcpyHostToDevice));
        delete[] resultArray;

        cout << "PREP--graph data in device memory are now ready" << endl;
    }

  private:
    int vert_num;
    int edge_num;

    // Allocate memory for data in GPU.
    void CudaMallocData() {
        // graph
        CUDA_ERROR(cudaMalloc(&csr_v, sizeof(Vertex) * (vert_num + 1 )));
        CUDA_ERROR(cudaMalloc(&csr_e, sizeof(Vertex) * edge_num));
        CUDA_ERROR(cudaMalloc(&csr_w, sizeof(ValueType) * edge_num));
        
        CUDA_ERROR(cudaMalloc(&distance, sizeof(ValueType) * vert_num ));  
              
        CUDA_ERROR(cudaMalloc(&active_vert, sizeof(Vertex) * vert_num));
        CUDA_ERROR(cudaMalloc(&active_vert_num, sizeof(Vertex) * 1 ));
        CUDA_ERROR(cudaMalloc(&isactive, sizeof(Vertex) * vert_num ));

        CUDA_ERROR(cudaMalloc(&iteration_num, sizeof(int)));
	   CUDA_ERROR(cudaMalloc(&iteration_act_num, sizeof(int)*1000));
    }

    // using array to replace vector
    template<typename ValueTypex>
    ValueTypex *vec_to_arr(vector<ValueTypex> vec) {

        Vertex vec_size = vec.size();
        ValueTypex *ptr = new ValueTypex[vec_size];

        for (Vertex i = 0; i < vec_size; i++) {
            ptr[i] = vec[i];
        }

        return ptr;
    }

    // Release memory for data in GPU.
    void CudaFreeData() {
        //

        if (csr_e)
            CUDA_ERROR(cudaFree(csr_e));
        if (csr_w)
            CUDA_ERROR(cudaFree(csr_w));
        if (csr_v)
            CUDA_ERROR(cudaFree(csr_v));
        if (active_vert)
            CUDA_ERROR(cudaFree(active_vert));
        if (active_vert_num)
            CUDA_ERROR(cudaFree(active_vert_num));
        if (distance)
            CUDA_ERROR(cudaFree(distance));
        if (isactive)
            CUDA_ERROR(cudaFree(isactive));
        if (iteration_num) CUDA_ERROR(cudaFree(iteration_num));
	   if (iteration_act_num) CUDA_ERROR(cudaFree(iteration_act_num));
    }
};
