#include "graph.h"

class DeviceMemory {
public:
    Vertex *csr_v;
    Vertex *csr_e;
    Vertex *csr_o;
    Vertex *csr_r;
    Vertex *degree;
    Vertex *active_vert;
    Vertex * active_verts_numStream;

    ValueType *pagerankStream; //i.e., reserve
    ValueType *residualStream;
    ValueType *messagesStream;
    bool      *isactive;

    int *iteration_num;
    int *iteration_act_num;

    int *acv;

    //size_t pitch; // 存储实际行大小的变量

    DeviceMemory(int _v_num, int _r_num, int _e_num, int _m_num)
            : vert_num(_v_num), rule_num(_r_num), edge_num(_e_num) ,max_step(_m_num){
        vaddr = vert_num + rule_num;

        csr_v = NULL;
        csr_e = NULL;
        csr_o = NULL;
        csr_r = NULL;
        degree = NULL;
        active_vert = NULL;

        active_verts_numStream = NULL;
        pagerankStream = NULL;
        residualStream = NULL;
        messagesStream = NULL;
        isactive = NULL;

        iteration_num = NULL;
        iteration_act_num =NULL;

        acv = NULL;


        CudaMallocData();
        //cout << "INIT--class DeviceMemory is constructed" << endl;
    }

    ~DeviceMemory() {
        CudaFreeData();
        //cout << "CLEAR--class DeviceMemory is destroyed" << endl;
    }

    // Copy graph data from host to device.
    void CudaMemcpyGraph(Graph &graph) {

        Vertex *resultArray = vec_to_arr(graph.csr_e);
        CUDA_ERROR(cudaMemcpy(csr_e, resultArray, sizeof(Vertex) * edge_num,
                              cudaMemcpyHostToDevice));
        delete[] resultArray;

        resultArray = vec_to_arr(graph.csr_v);
        CUDA_ERROR(cudaMemcpy(csr_v, resultArray, sizeof(Vertex) * (vaddr + 1),
                              cudaMemcpyHostToDevice));
        delete[] resultArray;

        resultArray = vec_to_arr(graph.csr_o);
        CUDA_ERROR(cudaMemcpy(csr_o, resultArray, sizeof(Vertex) * (max_step + 1),
                              cudaMemcpyHostToDevice));
        delete[] resultArray;

        resultArray = vec_to_arr(graph.csr_r);
        CUDA_ERROR(cudaMemcpy(csr_r, resultArray, sizeof(Vertex) * rule_num,
                              cudaMemcpyHostToDevice));
        delete[] resultArray;

        resultArray = vec_to_arr(graph.degree);
        CUDA_ERROR(cudaMemcpy(degree, resultArray, sizeof(Vertex) * vert_num,
                              cudaMemcpyHostToDevice));
        delete[] resultArray;



        //cout << "PREP--graph data in device memory are now ready" << endl;
    }

private:
    int vert_num;
    int rule_num;
    int edge_num;
    int vaddr;
    int max_step;

    // Allocate memory for data in GPU.
    void CudaMallocData() {
        // graph
        CUDA_ERROR(cudaMalloc(&csr_e, sizeof(Vertex) * edge_num));
        CUDA_ERROR(cudaMalloc(&csr_v, sizeof(Vertex) * (vaddr + 1 )));

        CUDA_ERROR(cudaMalloc(&csr_o, sizeof(Vertex) * (max_step + 1)));
        CUDA_ERROR(cudaMalloc(&csr_r, sizeof(Vertex) * rule_num));
        CUDA_ERROR(cudaMalloc(&degree, sizeof(Vertex) * vert_num));


        CUDA_ERROR(cudaMalloc(&active_vert, sizeof(Vertex) * vert_num));

        CUDA_ERROR(cudaMalloc(&active_verts_numStream, sizeof(Vertex) * 1 ));
        CUDA_ERROR(cudaMalloc(&pagerankStream, sizeof(Vertex) * vaddr ));
        CUDA_ERROR(cudaMalloc(&residualStream, sizeof(Vertex) * vaddr ));
        CUDA_ERROR(cudaMalloc(&messagesStream, sizeof(Vertex) * vaddr ));
        CUDA_ERROR(cudaMalloc(&isactive, sizeof(Vertex) * vaddr ));

        CUDA_ERROR(cudaMalloc(&iteration_num, sizeof(int)));
        CUDA_ERROR(cudaMalloc(&acv, sizeof(int)));
        CUDA_ERROR(cudaMalloc(&iteration_act_num, sizeof(int)*1000));
    }

    // using array to replace vector
    Vertex *vec_to_arr(vector<Vertex> vec) {

        Vertex vec_size = vec.size();
        Vertex *ptr = new Vertex[vec_size];

        for (Vertex i = 0; i < vec_size; i++) {
            ptr[i] = vec[i];
        }

        return ptr;
    }

    // Release memory for data in GPU.
    void CudaFreeData() {
        //
        if (csr_v)
            CUDA_ERROR(cudaFree(csr_v));
        if (csr_e)
            CUDA_ERROR(cudaFree(csr_e));
        if (csr_o)
            CUDA_ERROR(cudaFree(csr_o));
        if (csr_r)
            CUDA_ERROR(cudaFree(csr_r));
        if (degree)
            CUDA_ERROR(cudaFree(degree));

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
        if (acv) CUDA_ERROR(cudaFree(acv));
        if (iteration_act_num) CUDA_ERROR(cudaFree(iteration_act_num));

    }
};
