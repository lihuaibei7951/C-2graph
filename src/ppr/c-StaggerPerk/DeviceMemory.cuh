#include "Graph.h"

class DeviceMemory {
  public:
    Vertex *csr_v;
    Vertex *csr_e;
    Vertex *degree;
    Vertex *order;
    
    Vertex *active_vert1;
    Vertex * active_verts_numStream1;
	ValueType *pagerankStream1; //i.e., reserve
	ValueType *residualStream1;
	ValueType *messagesStream1;
	bool      *isactive1;

    Vertex *active_vert2;
    Vertex * active_verts_numStream2;
	ValueType *pagerankStream2; //i.e., reserve
	ValueType *residualStream2;
	ValueType *messagesStream2;
	bool      *isactive2;
    
    //size_t pitch; // 存储实际行大小的变量

    DeviceMemory(int _v_num, int _r_num, int _e_num, int _m_num)
        : vert_num(_v_num), rule_num(_r_num), edge_num(_e_num) ,max_step(_m_num){
        vaddr = vert_num + rule_num;

        csr_v = NULL;
        csr_e = NULL;
        degree = NULL;
        order = NULL;
        
        active_vert1 = NULL;
        active_verts_numStream1 = NULL;
        pagerankStream1 = NULL;
        residualStream1 = NULL;
        messagesStream1 = NULL;
        isactive1 = NULL;
        
        active_vert2 = NULL;
        active_verts_numStream2 = NULL;
        pagerankStream2 = NULL;
        residualStream2 = NULL;
        messagesStream2 = NULL;
        isactive2 = NULL;
        

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

        resultArray = vec_to_arr(graph.degree);
        CUDA_ERROR(cudaMemcpy(degree, resultArray, sizeof(Vertex) * vert_num,
                              cudaMemcpyHostToDevice));
        delete[] resultArray;

        resultArray = vec_to_arr(graph.order);
        CUDA_ERROR(cudaMemcpy(order, resultArray, sizeof(Vertex) * vaddr,
                              cudaMemcpyHostToDevice));
        delete[] resultArray;

        resultArray = vec_to_arr(graph.csr_v);
        CUDA_ERROR(cudaMemcpy(csr_v, resultArray, sizeof(Vertex) * (vaddr + 1),
                              cudaMemcpyHostToDevice));
        delete[] resultArray;

        cout << "PREP--graph data in device memory are now ready" << endl;
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
        CUDA_ERROR(cudaMalloc(&degree, sizeof(Vertex) * vert_num));
        CUDA_ERROR(cudaMalloc(&order, sizeof(Vertex) * vaddr));
        CUDA_ERROR(cudaMalloc(&csr_v, sizeof(Vertex) * (vaddr + 1 )));
        
        CUDA_ERROR(cudaMalloc(&active_vert1, sizeof(Vertex) * vert_num));
        CUDA_ERROR(cudaMalloc(&active_verts_numStream1, sizeof(Vertex) * 1 ));
        CUDA_ERROR(cudaMalloc(&pagerankStream1, sizeof(Vertex) * vaddr ));
        CUDA_ERROR(cudaMalloc(&residualStream1, sizeof(Vertex) * vaddr ));
        CUDA_ERROR(cudaMalloc(&messagesStream1, sizeof(Vertex) * vaddr ));
        CUDA_ERROR(cudaMalloc(&isactive1, sizeof(Vertex) * (vert_num + rule_num*(max_step+1)) ));

        CUDA_ERROR(cudaMalloc(&active_vert2, sizeof(Vertex) * vert_num));
        CUDA_ERROR(cudaMalloc(&active_verts_numStream2, sizeof(Vertex) * 1 ));
        CUDA_ERROR(cudaMalloc(&pagerankStream2, sizeof(Vertex) * vaddr ));
        CUDA_ERROR(cudaMalloc(&residualStream2, sizeof(Vertex) * vaddr ));
        CUDA_ERROR(cudaMalloc(&messagesStream2, sizeof(Vertex) * vaddr ));
        CUDA_ERROR(cudaMalloc(&isactive2, sizeof(Vertex) * (vert_num + rule_num*(max_step+1)) ));
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

        if (csr_e)
            CUDA_ERROR(cudaFree(csr_e));
        if (degree)
            CUDA_ERROR(cudaFree(degree));
        if (order)
            CUDA_ERROR(cudaFree(order));
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
        if (messagesStream1)
            CUDA_ERROR(cudaFree(messagesStream1));
        if (isactive1)
            CUDA_ERROR(cudaFree(isactive1));

        if (active_vert2)
            CUDA_ERROR(cudaFree(active_vert2));
        if (active_verts_numStream2)
            CUDA_ERROR(cudaFree(active_verts_numStream2));
        if (pagerankStream2)
            CUDA_ERROR(cudaFree(pagerankStream2));
        if (residualStream2)
            CUDA_ERROR(cudaFree(residualStream2));
        if (messagesStream2)
            CUDA_ERROR(cudaFree(messagesStream2));
        if (isactive2)
            CUDA_ERROR(cudaFree(isactive2));
    }
};
