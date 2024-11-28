#include "Graph.h"

class DeviceMemory {
public:
    Vertex *csr_v;
    Vertex *csr_e;
    ValueType *csr_w;
    Vertex *active_vert;
    Vertex * active_verts_numStream;

    ValueType *pagerankStream; //i.e., reserve
    ValueType *residualStream;
    ValueType *messagesStream;
    bool      *isactive;
    int *iteration_num;
    int *iteration_act_num;



    //size_t pitch; // 存储实际行大小的变量

    DeviceMemory(int _v_num, int _e_num)
            : vert_num(_v_num), edge_num(_e_num) {

        csr_v = NULL;
        csr_e = NULL;
        csr_w = NULL;
        active_vert = NULL;

        active_verts_numStream = NULL;
        pagerankStream = NULL;
        residualStream = NULL;
        messagesStream = NULL;
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

        resultArray = vec_to_arr(graph.csr_v);
        CUDA_ERROR(cudaMemcpy(csr_v, resultArray, sizeof(Vertex) * (vert_num + 1),
                              cudaMemcpyHostToDevice));
        delete[] resultArray;

        ValueType *resul = vec_to_arr(graph.csr_w);
        CUDA_ERROR(cudaMemcpy(csr_w, resul, sizeof(ValueType) * edge_num,
                              cudaMemcpyHostToDevice));
        delete[] resul;

        cout << "PREP--graph data in device memory are now ready" << endl;
    }

private:
    int vert_num;
    int edge_num;

    // Allocate memory for data in GPU.
    void CudaMallocData() {
        // graph
        CUDA_ERROR(cudaMalloc(&csr_e, sizeof(Vertex) * edge_num));
        CUDA_ERROR(cudaMalloc(&csr_w, sizeof(ValueType) * edge_num));
        CUDA_ERROR(cudaMalloc(&csr_v, sizeof(Vertex) * (vert_num + 1 )));
        CUDA_ERROR(cudaMalloc(&active_vert, sizeof(Vertex) * vert_num));

        CUDA_ERROR(cudaMalloc(&active_verts_numStream, sizeof(Vertex) * 1 ));
        CUDA_ERROR(cudaMalloc(&pagerankStream, sizeof(ValueType) * vert_num ));
        CUDA_ERROR(cudaMalloc(&residualStream, sizeof(ValueType) * vert_num ));
        CUDA_ERROR(cudaMalloc(&messagesStream, sizeof(ValueType) * vert_num ));
        CUDA_ERROR(cudaMalloc(&isactive, sizeof(bool) * vert_num ));

        CUDA_ERROR(cudaMalloc(&iteration_num, sizeof(int)));
        CUDA_ERROR(cudaMalloc(&iteration_act_num, sizeof(int)*1000));
    }

    // using array to replace vector
    template<typename VA>
    VA *vec_to_arr(vector<VA> vec) {

        Vertex vec_size = vec.size();
        VA *ptr = new VA[vec_size];

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
        if (csr_v)
            CUDA_ERROR(cudaFree(csr_v));
        if (csr_w)
            CUDA_ERROR(cudaFree(csr_w));
        if (active_vert)
            CUDA_ERROR(cudaFree(active_vert));
        if (active_verts_numStream)
            CUDA_ERROR(cudaFree(active_verts_numStream));
        if (pagerankStream)
            CUDA_ERROR(cudaFree(pagerankStream));
        if (residualStream)
            CUDA_ERROR(cudaFree(residualStream));
        if (messagesStream)
            CUDA_ERROR(cudaFree(messagesStream));
        if (isactive)
            CUDA_ERROR(cudaFree(isactive));
        if (iteration_num) CUDA_ERROR(cudaFree(iteration_num));
        if (iteration_act_num) CUDA_ERROR(cudaFree(iteration_act_num));
    }
};
