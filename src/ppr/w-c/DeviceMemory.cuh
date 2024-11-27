#include "Graph.h"

class DeviceMemory {
public:
    Vertex *csr_ov;
    Vertex *csr_v;
    Vertex *csr_e;
    Vertex *csr_idx;
    ValueType *csr_w; //i.e., reserve

    ValueType *pagerankStream; //i.e., reserve
    ValueType *residualStream;
    ValueType *messagesStream;

    bool *isactive;
    Vertex *active_vert1;
    Vertex *active_vert2;
    Vertex *active_vert;
    Vertex *active_vert_num1;
    Vertex *active_vert_num2;
    Vertex *active_vert_num;


    int *iteration_num;
    int *iteration_act_num;


    ValueType *active_w1;
    ValueType *active_w2;

    //size_t pitch; // 存储实际行大小的变量

    DeviceMemory(int _v_num,int _r_num, int _e_num,int _w_num)
            : vert_num(_v_num),rule_num(_r_num), edge_num(_e_num),w_num(_w_num) {

        csr_v = NULL;
        csr_ov = NULL;
        csr_e = NULL;
        csr_idx = NULL;
        csr_w = NULL;
        pagerankStream = NULL;
        residualStream = NULL;
        messagesStream = NULL;

        active_vert1 = NULL;
        active_vert2 = NULL;
        active_vert = NULL;
        active_vert_num1 = NULL;
        active_vert_num2 = NULL;
        active_vert = NULL;

        isactive = NULL;
        iteration_num = NULL;
        iteration_act_num = NULL;
        add_num = vert_num + rule_num;


        active_w1 = NULL;
        active_w2 = NULL;

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

        resultArray = vec_to_arr(graph.csr_idx);
        CUDA_ERROR(cudaMemcpy(csr_idx, resultArray, sizeof(Vertex) * edge_num,
                              cudaMemcpyHostToDevice));
        delete[] resultArray;

        resultArray = vec_to_arr(graph.csr_v);
        CUDA_ERROR(cudaMemcpy(csr_v, resultArray, sizeof(Vertex) * (add_num + 1),
                              cudaMemcpyHostToDevice));
        delete[] resultArray;

        resultArray = vec_to_arr(graph.csr_ov);
        CUDA_ERROR(cudaMemcpy(csr_ov, resultArray, sizeof(Vertex) * (vert_num + 1),
                              cudaMemcpyHostToDevice));
        delete[] resultArray;

        

        ValueType *result = vec_to_arr(graph.csr_w);
        CUDA_ERROR(cudaMemcpy(csr_w, result, sizeof(ValueType) * w_num,
                              cudaMemcpyHostToDevice));
        delete[] result;



        cout << "PREP--graph data in device memory are now ready" << endl;
    }

private:
    int vert_num,rule_num,add_num;
    int edge_num,w_num;

    // Allocate memory for data in GPU.
    void CudaMallocData() {
        // graph
        CUDA_ERROR(cudaMalloc(&csr_v, sizeof(Vertex) * (add_num + 1)));

        CUDA_ERROR(cudaMalloc(&csr_ov, sizeof(Vertex) * (vert_num + 1)));
        CUDA_ERROR(cudaMalloc(&csr_e, sizeof(Vertex) * edge_num));
        CUDA_ERROR(cudaMalloc(&csr_idx, sizeof(Vertex) * edge_num));
        
        CUDA_ERROR(cudaMalloc(&csr_w, sizeof(ValueType) * w_num));

        CUDA_ERROR(cudaMalloc(&pagerankStream, sizeof(Vertex) * vert_num ));
        CUDA_ERROR(cudaMalloc(&residualStream, sizeof(Vertex) * vert_num ));
        CUDA_ERROR(cudaMalloc(&messagesStream, sizeof(Vertex) * vert_num ));

        CUDA_ERROR(cudaMalloc(&active_vert1, sizeof(Vertex) * edge_num*3));
        CUDA_ERROR(cudaMalloc(&active_vert2, sizeof(Vertex) * edge_num*3));

        CUDA_ERROR(cudaMalloc(&active_vert, sizeof(Vertex) * vert_num));

        CUDA_ERROR(cudaMalloc(&active_w1, sizeof(ValueType) * edge_num));
        CUDA_ERROR(cudaMalloc(&active_w2, sizeof(ValueType) * edge_num));

        CUDA_ERROR(cudaMalloc(&active_vert_num1, sizeof(Vertex) * 1));
        CUDA_ERROR(cudaMalloc(&active_vert_num2, sizeof(Vertex) * 1));

        CUDA_ERROR(cudaMalloc(&active_vert_num, sizeof(Vertex) * 1));
        CUDA_ERROR(cudaMalloc(&isactive, sizeof(bool) * vert_num));

        CUDA_ERROR(cudaMalloc(&iteration_num, sizeof(int)));
        CUDA_ERROR(cudaMalloc(&iteration_act_num, sizeof(int) * 2000));


    }

    // using array to replace vector
    template<class ValueTypex>
    ValueTypex *vec_to_arr(vector< ValueTypex > vec) {

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
        if (csr_idx)
            CUDA_ERROR(cudaFree(csr_idx));
        if (csr_w)
            CUDA_ERROR(cudaFree(csr_w));
        if (csr_v)
            CUDA_ERROR(cudaFree(csr_v));
        if (csr_ov)
            CUDA_ERROR(cudaFree(csr_ov));
        if (active_vert1)
            CUDA_ERROR(cudaFree(active_vert1));
        if (active_vert2)
            CUDA_ERROR(cudaFree(active_vert2));
        if (active_vert)
            CUDA_ERROR(cudaFree(active_vert));
        if (active_w1)
            CUDA_ERROR(cudaFree(active_w1));
        if (active_w2)
            CUDA_ERROR(cudaFree(active_w2));
        if (active_vert_num1)
            CUDA_ERROR(cudaFree(active_vert_num1));
        if (active_vert_num2)
            CUDA_ERROR(cudaFree(active_vert_num2));
        if (active_vert_num)
            CUDA_ERROR(cudaFree(active_vert_num));
        if (isactive)
            CUDA_ERROR(cudaFree(isactive));
        if (iteration_num)
            CUDA_ERROR(cudaFree(iteration_num));
        if (iteration_act_num)
            CUDA_ERROR(cudaFree(iteration_act_num));
        if (pagerankStream)
            CUDA_ERROR(cudaFree(pagerankStream));
        if (residualStream)
            CUDA_ERROR(cudaFree(residualStream));
        if (messagesStream)
            CUDA_ERROR(cudaFree(messagesStream));
    }
};
