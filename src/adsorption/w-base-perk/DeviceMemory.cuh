#include "Graph.h"

class DeviceMemory {
public:
    Vertex *csr_v;
    Vertex *csr_e;
    ValueType *csr_w;

    Vertex *active_vert1;
    Vertex * active_verts_numStream1;
    ValueType *pagerankStream1; //i.e., reserve
    ValueType *residualStream1;

    Vertex *active_vert2;
    Vertex * active_verts_numStream2;
    ValueType *pagerankStream2; //i.e., reserve
    ValueType *residualStream2;

    ValueType *messagesStream;
    bool      *isactive;

    //size_t pitch; // 存储实际行大小的变量

    DeviceMemory(int _v_num, int _e_num)
            : vert_num(_v_num), edge_num(_e_num) {

        csr_v = NULL;
        csr_e = NULL;

        active_vert1 = NULL;
        active_verts_numStream1 = NULL;
        pagerankStream1 = NULL;
        residualStream1 = NULL;


        active_vert2 = NULL;
        active_verts_numStream2 = NULL;
        pagerankStream2 = NULL;
        residualStream2 = NULL;

        messagesStream = NULL;
        isactive = NULL;


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

        ValueType *resultArray1 = vec_to_arr(graph.csr_w);
        CUDA_ERROR(cudaMemcpy(csr_w, resultArray1, sizeof(ValueType) * edge_num,
                              cudaMemcpyHostToDevice));
        delete[] resultArray1;

        cout << "PREP--graph data in device memory are now ready" << endl;
    }

private:
    int vert_num;
    int edge_num;

    // Allocate memory for data in GPU.
    void CudaMallocData() {
        // graph
        CUDA_ERROR(cudaMalloc(&csr_e, sizeof(Vertex) * edge_num));
        CUDA_ERROR(cudaMalloc(&csr_v, sizeof(Vertex) * (vert_num + 1 )));
        CUDA_ERROR(cudaMalloc(&csr_w, sizeof(ValueType) * edge_num));

        CUDA_ERROR(cudaMalloc(&active_vert1, sizeof(Vertex) * vert_num));
        CUDA_ERROR(cudaMalloc(&active_verts_numStream1, sizeof(Vertex) * 1 ));
        CUDA_ERROR(cudaMalloc(&pagerankStream1, sizeof(Vertex) * vert_num ));
        CUDA_ERROR(cudaMalloc(&residualStream1, sizeof(Vertex) * vert_num ));



        CUDA_ERROR(cudaMalloc(&active_vert2, sizeof(Vertex) * vert_num));
        CUDA_ERROR(cudaMalloc(&active_verts_numStream2, sizeof(Vertex) * 1 ));
        CUDA_ERROR(cudaMalloc(&pagerankStream2, sizeof(Vertex) * vert_num ));
        CUDA_ERROR(cudaMalloc(&residualStream2, sizeof(Vertex) * vert_num ));

        CUDA_ERROR(cudaMalloc(&messagesStream, sizeof(Vertex) * vert_num ));
        CUDA_ERROR(cudaMalloc(&isactive, sizeof(Vertex) * vert_num  ));
    }

    // using array to replace vector
    template<typename T>
    T *vec_to_arr(vector<T> vec) {

        Vertex vec_size = vec.size();
        T *ptr = new T[vec_size];

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

        if (active_vert1)
            CUDA_ERROR(cudaFree(active_vert1));
        if (active_verts_numStream1)
            CUDA_ERROR(cudaFree(active_verts_numStream1));
        if (pagerankStream1)
            CUDA_ERROR(cudaFree(pagerankStream1));
        if (residualStream1)
            CUDA_ERROR(cudaFree(residualStream1));
        if (messagesStream)
            CUDA_ERROR(cudaFree(messagesStream));
        if (isactive)
            CUDA_ERROR(cudaFree(isactive));

        if (active_vert2)
            CUDA_ERROR(cudaFree(active_vert2));
        if (active_verts_numStream2)
            CUDA_ERROR(cudaFree(active_verts_numStream2));
        if (pagerankStream2)
            CUDA_ERROR(cudaFree(pagerankStream2));
        if (residualStream2)
            CUDA_ERROR(cudaFree(residualStream2));
    }
};
