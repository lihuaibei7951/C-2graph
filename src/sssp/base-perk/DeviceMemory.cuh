#include "Graph.h"

class DeviceMemory {
public:
    Vertex *csr_v;
    Vertex *csr_e;
    ValueType *csr_w; //i.e., reserve

    ValueType *distance1; //i.e., reserve
    Vertex *active_vert1;
    Vertex * active_vert_num1;

    ValueType *distance2; //i.e., reserve
    Vertex *active_vert2;
    Vertex * active_vert_num2;

    bool      *isactive1;
    bool      *isactive2;

    //size_t pitch; // 存储实际行大小的变量

    DeviceMemory(int _v_num, int _e_num)
            : vert_num(_v_num), edge_num(_e_num) {

        csr_v = NULL;
        csr_e = NULL;
        csr_w = NULL;

        distance1 = NULL;
        active_vert1 = NULL;
        active_vert_num1 = NULL;
        isactive1 = NULL;
        isactive2 = NULL;

        distance2 = NULL;
        active_vert2 = NULL;
        active_vert_num2 = NULL;


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

        CUDA_ERROR(cudaMalloc(&distance1, sizeof(ValueType) * vert_num ));
        CUDA_ERROR(cudaMalloc(&active_vert1, sizeof(Vertex) * vert_num));
        CUDA_ERROR(cudaMalloc(&active_vert_num1, sizeof(Vertex) * 1 ));
        CUDA_ERROR(cudaMalloc(&isactive1, sizeof(bool) * vert_num ));
        CUDA_ERROR(cudaMalloc(&isactive2, sizeof(bool) * vert_num ));

        CUDA_ERROR(cudaMalloc(&distance2, sizeof(ValueType) * vert_num ));
        CUDA_ERROR(cudaMalloc(&active_vert2, sizeof(Vertex) * vert_num));
        CUDA_ERROR(cudaMalloc(&active_vert_num2, sizeof(Vertex) * 1 ));
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

        if (active_vert1)
            CUDA_ERROR(cudaFree(active_vert1));
        if (active_vert_num1)
            CUDA_ERROR(cudaFree(active_vert_num1));
        if (distance1)
            CUDA_ERROR(cudaFree(distance1));
        if (isactive1)
            CUDA_ERROR(cudaFree(isactive1));
        if (isactive2)
            CUDA_ERROR(cudaFree(isactive2));

        if (active_vert2)
            CUDA_ERROR(cudaFree(active_vert2));
        if (active_vert_num2)
            CUDA_ERROR(cudaFree(active_vert_num2));
        if (distance2)
            CUDA_ERROR(cudaFree(distance2));
    }
};
