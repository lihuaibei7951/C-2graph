#include <time.h>
#include "Util.cuh"
#include "DeviceMemory.cuh"

__device__ volatile int g_mutex1;
__device__ volatile int g_mutex2;
__device__ volatile int g_mutex3;
__device__ volatile int g_mutex4;
__device__ volatile int g_mutex5;
__device__ volatile int g_mutex6;
//原子操作重写
__device__ static float atomicMin(float *address, float val) {
    int *address_as_i = (int *) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

//kernel_ update
template<typename ValueType>
__global__ void CalcuPPR(const Vertex *csr_v, const Vertex *csr_e, const ValueType *csr_w,
                         ValueType *pagerank, ValueType *residual, ValueType *messages,
                          const Vertex *csr_ov, const Vertex *csr_idx,
                          Vertex *active_vert1,Vertex *active_vert2, Vertex *active_vert,ValueType *active_w1,ValueType *active_w2,
                          Vertex *active_vert_num1,Vertex *active_vert_num2,Vertex *active_vert_num, bool *isactive,
                          const Vertex vert_num, Vertex source, Vertex *iteration_id, Vertex *iteration_num,
                          Vertex *iteration_act_num,ValueType alpha,ValueType rmax);

int main(int argc, char **argv) {
    // Initialize graph data in host & device memory
    cudaFree(0);
    // 获取命令行参数
    std::string dir = argv[1];
    int source = atoi(argv[2]);

    // 获取，csr_v ,csr_e ,v_r,degree,order;
    Graph graph(dir);

    DeviceMemory device_memory(graph.vert_num, graph.rule_num,graph.edge_num,graph.w_num);

    device_memory.CudaMemcpyGraph(graph);
    std::cout << "test for study how to use cuda" << endl;
    int vert_num = graph.vert_num;
    int add_num = graph.add_num;
    int rule_num = graph.rule_num;
    int edge_num = graph.edge_num;


    int *iteration_id;
    iteration_id = NULL;
    cudaMalloc(&iteration_id, sizeof(int) * 2000);

    int *iteration_num = new int[1];
    int *ac1 = new int[1];
    int *ac2 = new int[1];
    int *iteration_act_num = new int[2000];
    ValueType alpha = 0.2f;
    ValueType rmax =0.001f/(graph.vert_num);
    cout << "\n==================== SSSP with FORWARD PUSH starts ============" << endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    int cnt = 0;

    while (1) {
cudaDeviceSynchronize();
        CalcuPPR<ValueType><<<MAX_BLOCKS_NUM, THREADS_PER_BLOCK>>>(
                    device_memory.csr_v, device_memory.csr_e, device_memory.csr_w,
                    device_memory.pagerankStream,device_memory.residualStream,device_memory.messagesStream,
                    device_memory.csr_ov, device_memory.csr_idx,
                    device_memory.active_vert1,device_memory.active_vert2,device_memory.active_vert,
                    device_memory.active_w1,device_memory.active_w2,
                    device_memory.active_vert_num1,device_memory.active_vert_num2, device_memory.active_vert_num,
                    device_memory.isactive,
                    vert_num, source, iteration_id, device_memory.iteration_num, device_memory.iteration_act_num, alpha, rmax);


        cnt++;
        if(cnt % 10 ==0){
            cudaDeviceSynchronize();
        }
        if (cnt == 10) {
        	cudaDeviceSynchronize();
            break;
        }

    }


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    CUDA_ERROR(cudaMemcpy(iteration_num, device_memory.iteration_num, sizeof(int) * 1, cudaMemcpyDeviceToHost));
    
    cout << "flag 已设置成 -1  终止条件以满足		iteration_num：" << iteration_num[0] << endl;
    CUDA_ERROR(cudaMemcpy(iteration_act_num, device_memory.iteration_act_num,sizeof(int) * 200, cudaMemcpyDeviceToHost));
    cout << "0	act_num：1" << endl;
    for (int i = 1; iteration_act_num[i] != 0; i++) {
        cout << i << "	act_num：" << iteration_act_num[i] << endl;
        if (i > 198) break;
    }
    float runtime = 0;
    cudaEventElapsedTime(&runtime, start, stop);
    cout << "gpu runtime: " << runtime / 1000.0 << " seconds" << endl;
    cout << "源顶点source = " << source - 1 << endl;

    return 0;
}


template<typename ValueType>
__global__ void CalcuPPR(const Vertex *csr_v, const Vertex *csr_e, const ValueType *csr_w,
                         ValueType *pagerank, ValueType *residual, ValueType *messages,
                         const Vertex *csr_ov, const Vertex *csr_idx,
                         Vertex *active_vert1,Vertex *active_vert2, Vertex *active_vert,ValueType *active_w1,ValueType *active_w2,
                         Vertex *active_vert_num1,Vertex *active_vert_num2,Vertex *active_vert_num, bool *isactive,
                         const Vertex vert_num, Vertex source, Vertex *iteration_id, Vertex *iteration_num,
                         Vertex *iteration_act_num,ValueType alpha,ValueType rmax) {
    size_t thread_id = threadIdx.x;
    size_t schedule_offset_init = blockDim.x * blockIdx.x;
    size_t vid = 0;

    while (schedule_offset_init < vert_num) {

        vid = schedule_offset_init + thread_id;

        if (vid < vert_num) {
            pagerank[vid] = 0;
            residual[vid] = 0;
            isactive[vid] = false;
            messages[vid] = 0;
        }
        if (vid < 2000) {
            iteration_id[vid] = vid;
        }
        schedule_offset_init += blockDim.x * gridDim.x;
    }

    //prepare for iteration
    size_t global_id = thread_id + blockDim.x * blockIdx.x;
    if (global_id == 0) {
        active_vert[0] = source;
        pagerank[source] = alpha * residual[source];
        residual[source] = 1;
        *active_vert_num1 = 0;
        *active_vert_num2 = 0;
        *active_vert_num = 1;
        g_mutex1 = 0;
        g_mutex2 = 0;
        g_mutex4 = 0;
        g_mutex5 = 0;
        g_mutex6 = 0;
    }

    __threadfence();
    if (threadIdx.x == 0) {
        atomicAdd((int*) &g_mutex3, 1);
        while ((g_mutex3 == 0) || (g_mutex3 % gridDim.x) ) {}
    }
    __syncthreads();
    int l_iteration_id = 0;
    int total_avtive_num;
    thread_id = threadIdx.x;//当前块内的线程id
    size_t lane_id = thread_id % THREADS_PER_WARP;
    size_t warp_id = thread_id / THREADS_PER_WARP;

    typedef cub::BlockScan<int, THREADS_PER_BLOCK> BlockScan;
    __shared__ typename BlockScan::TempStorage block_temp_storage;

    volatile __shared__ int comm[THREADS_PER_BLOCK / THREADS_PER_WARP][3];//[256/32][3]第一维是多少个warp，256/32=8，8个warp，
    volatile __shared__ ValueType commr[THREADS_PER_BLOCK / THREADS_PER_WARP];//每个warp对应一个值
    volatile __shared__ int commi[THREADS_PER_BLOCK / THREADS_PER_WARP][2];//每个warp对应3个值
    volatile __shared__ int comm2[THREADS_PER_BLOCK]; //一维数组大小256，int
    volatile __shared__ int commd2[THREADS_PER_BLOCK]; //out-degree
    volatile __shared__ ValueType commr2[THREADS_PER_BLOCK];
    volatile __shared__ int commi2[THREADS_PER_BLOCK][2];//每个warp对应3个值

    while ((*active_vert_num1 >0||*active_vert_num2 >0||*active_vert_num>0)&&l_iteration_id<2000) {
        l_iteration_id += 1;
        __syncthreads();
        __threadfence();
        if (threadIdx.x == 0) {
            atomicAdd((int *) &g_mutex6, 1);
            while (g_mutex6 < gridDim.x * iteration_id[l_iteration_id]) {}
            iteration_num[0] = l_iteration_id;
            if(l_iteration_id<2000)	iteration_act_num[l_iteration_id]= (*active_vert_num)+(*active_vert_num1)/2+(*active_vert_num2)/2;
        }
        __syncthreads();
        if(l_iteration_id%2==1){
            //先顶点遍历
            __syncthreads();
            total_avtive_num=*active_vert_num;

            __syncthreads();

            size_t schedule_offset = blockDim.x * blockIdx.x;
            size_t idx=0;
            int row_start=0, row_end=0,testx=0;
            int u=0, v=0, root=0, idxx=0;
            ValueType ru=0.0, msg=0.0; //dis value of u
            while (schedule_offset < total_avtive_num) {
                idx = schedule_offset + thread_id;
                if (idx < total_avtive_num) {
                    u = active_vert[idx];
                    //printf("%d",u);
                    ru =(1-alpha)* residual[u];
                    idxx=csr_ov[u];
                    residual[u] = 0;
                    row_start = csr_v[u]; //start offset of outgoing edges of "u" in "col_ind"
                    row_end = csr_v[u+1]; //end offset of outgoing edges of "u" in "col_ind" (exclusive)
                } else {
                    row_start = 0;
                    row_end = 0;
                }
                //while(1)
                while (__syncthreads_or((row_end-row_start)>=THREADS_PER_BLOCK)) {
                    if ((row_end-row_start) >= THREADS_PER_BLOCK) {
                        comm[0][0] = thread_id; //I (thread_id) want to process the active vertex assigned to me.
                    }
                    __syncthreads(); //all threads in one block vote to processing their own vertices

                    if (comm[0][0] == thread_id) {
                        comm[0][1] = row_start; //the vertx owned by me will be processed in this <1>-while loop.
                        comm[0][2] = row_end;
                        commr[0] = ru;//ru是u的残差
                        commi[0][1] = idxx;
                        row_start = row_end; //avoid processing this vertex repeatedly in <2>&<3>-while
                    }
                    __syncthreads(); //all threads are ready to process the selected vertex

                    size_t push_st = comm[0][1] + thread_id; //process the "push_st"-th outgoing edge at first.
                    size_t push_ed = comm[0][2];

                    // <1.1>-while: block-granularity-outgoing edges
                    while (__syncthreads_or(push_st<push_ed)) {
                        if (push_st < push_ed) {
                            v = csr_e[push_st]; //target vertex id
                            testx = commi[0][1] + csr_idx[push_st];
                            msg = commr[0]*csr_w[testx];
                            if(v<vert_num){
                                atomicAdd(messages + v, msg);
                                isactive[v] = true;
                            }else{
                                int cur = atomicAdd(active_vert_num1,2);
                                active_vert1[cur]=v;
                                active_vert1[cur+1]=testx;
                                active_w1[cur/2] = commr[0];
                            }
                        }
                        push_st += THREADS_PER_BLOCK; //until all outgoing edges of "u" have been processed
                    }
                } //until all source vertices with "todo_edges_num>=THREADS_PER_BLOCK" have been processed
                //while(2)
                while (__any_sync(FULL_MASK, (row_end-row_start)>=THREADS_PER_WARP)) {
                    if ((row_end-row_start) >= THREADS_PER_WARP) {
                        comm[warp_id][0] = lane_id; //threads in the "warp_id"-th warp try to vote
                    }
                    if (comm[warp_id][0] == lane_id) {
                        comm[warp_id][1] = row_start; //vertex owned by the "lane_id"-th thread in a warp is scheduled
                        comm[warp_id][2] = row_end;
                        commr[warp_id] = ru;
                        commi[warp_id][1] = idxx;
                        row_start = row_end; //avoid processing this vertex repeatedly in <3>-while
                    }
                    size_t push_st = comm[warp_id][1] + lane_id; //process the "push_st"-th outgoing edge at first.
                    size_t push_ed = comm[warp_id][2];

                    // <2.1>-while: warp-granularity-outgoing edges
                    while (__any_sync(FULL_MASK, push_st<push_ed)) {
                        if (push_st < push_ed) {
                            v = csr_e[push_st];
                            testx = commi[warp_id][1] + csr_idx[push_st];
                            msg = commr[warp_id]*csr_w[testx];
                            if(v<vert_num){
                                atomicAdd(messages + v, msg);
                                isactive[v] = true;
                            }else{
                                int cur = atomicAdd(active_vert_num1,2);
                                active_vert1[cur]=v;
                                active_vert1[cur+1]=testx;
                                active_w1[cur/2] = commr[warp_id];
                            }
                        }
                        push_st += THREADS_PER_WARP; //until all outgoing edges of "u" have been processed
                    }
                } //until all source vertices with "todo_edges_num>=THREADS_PER_WARP" have been processed

                //while(3) then, the out-degree of "u" is less than THREADS_PER_WARP(32)
                int thread_count = row_end - row_start;
                int deg = thread_count;
                int scatter = 0, total = 0;

                __syncthreads();
                BlockScan(block_temp_storage).ExclusiveSum(thread_count, scatter, total); //
                __syncthreads(); //there are "total" edges left in every block
                int progress = 0;
                while (progress < total) {
                    int remain = total - progress;
                    while (scatter<(progress+THREADS_PER_BLOCK) && (row_start<row_end)) {
                        comm2[scatter-progress] = row_start;//存U有的外邻居
                        commd2[scatter-progress] = deg; //
                        commr2[scatter-progress] = ru;
                        commi2[scatter - progress][1] = idxx;
                        scatter++;
                        row_start++;
                    }
                    __syncthreads();
                    int cur_batch_count = min(remain, (int)THREADS_PER_BLOCK); //how many threads are required?
                    if (thread_id < cur_batch_count) {
                        v = csr_e[comm2[thread_id]];
                        testx = commi2[thread_id][1] + csr_idx[comm2[thread_id]];
                        msg = (commr2[thread_id]) *csr_w[testx];
                        if(v<vert_num){
                            atomicAdd(messages + v, msg);
                            isactive[v] = true;
                        }else{
                            int cur = atomicAdd(active_vert_num1,2);
                            active_vert1[cur]=v;
                            active_vert1[cur+1]=testx;
                            active_w1[cur/2] =commr2[thread_id];
                        }
                    }
                    __syncthreads();
                    progress += THREADS_PER_BLOCK;
                }
                //schedule (blockDim.x * gridDim.x) active vertices per <0>-while loop
                schedule_offset += blockDim.x * gridDim.x;
            }
            __syncthreads();
            __threadfence();
            if (threadIdx.x == 0) {
                atomicAdd((int *) &g_mutex1, 1);
                while (g_mutex1 < gridDim.x * iteration_id[l_iteration_id]) {}
            }
            __syncthreads();
            //host 有一个操作，将 active_verts_num 设置为0
            *active_vert_num = 0;
            total_avtive_num= *active_vert_num1/2;
            __syncthreads();
            __threadfence();


            schedule_offset = blockDim.x * blockIdx.x;
            while (schedule_offset < total_avtive_num) {
                idx = schedule_offset + thread_id;
                if (idx < total_avtive_num) {
                    u = active_vert1[idx*2];
                    idxx=active_vert1[idx*2+1];
                    ru = active_w1[idx];
                    row_start = csr_v[u]; //start offset of outgoing edges of "u" in "col_ind"
                    row_end = csr_v[u+1]; //end offset of outgoing edges of "u" in "col_ind" (exclusive)
                } else {
                    row_start = 0;
                    row_end = 0;
                }
                //while(1)
                while (__syncthreads_or((row_end-row_start)>=THREADS_PER_BLOCK)) {
                    if ((row_end-row_start) >= THREADS_PER_BLOCK) {
                        comm[0][0] = thread_id; //I (thread_id) want to process the active vertex assigned to me.
                    }
                    __syncthreads(); //all threads in one block vote to processing their own vertices

                    if (comm[0][0] == thread_id) {
                        comm[0][1] = row_start; //the vertx owned by me will be processed in this <1>-while loop.
                        comm[0][2] = row_end;
                        commr[0] = ru;//ru是u的残差
                        commi[0][1] = idxx;
                        row_start = row_end; //avoid processing this vertex repeatedly in <2>&<3>-while
                    }
                    __syncthreads(); //all threads are ready to process the selected vertex

                    size_t push_st = comm[0][1] + thread_id; //process the "push_st"-th outgoing edge at first.
                    size_t push_ed = comm[0][2];

                    // <1.1>-while: block-granularity-outgoing edges
                    while (__syncthreads_or(push_st<push_ed)) {
                        if (push_st < push_ed) {
                            v = csr_e[push_st]; //target vertex id
                            testx = commi[0][1] + csr_idx[push_st];
                            msg = commr[0]*csr_w[testx];
                            if(v<vert_num){
                                atomicAdd(messages + v, msg);
                                isactive[v] = true;
                            }else{
                                int cur = atomicAdd(active_vert_num2,2);
                                active_vert2[cur]=v;
                                active_vert2[cur+1]=testx;
                                active_w2[cur/2] = commr[0];
                            }
                        }
                        push_st += THREADS_PER_BLOCK; //until all outgoing edges of "u" have been processed
                    }
                } //until all source vertices with "todo_edges_num>=THREADS_PER_BLOCK" have been processed
                //while(2)
                while (__any_sync(FULL_MASK, (row_end-row_start)>=THREADS_PER_WARP)) {
                    if ((row_end-row_start) >= THREADS_PER_WARP) {
                        comm[warp_id][0] = lane_id; //threads in the "warp_id"-th warp try to vote
                    }
                    if (comm[warp_id][0] == lane_id) {
                        comm[warp_id][1] = row_start; //vertex owned by the "lane_id"-th thread in a warp is scheduled
                        comm[warp_id][2] = row_end;
                        commr[warp_id] = ru;
                        commi[warp_id][1] = idxx;
                        row_start = row_end; //avoid processing this vertex repeatedly in <3>-while
                    }
                    size_t push_st = comm[warp_id][1] + lane_id; //process the "push_st"-th outgoing edge at first.
                    size_t push_ed = comm[warp_id][2];

                    // <2.1>-while: warp-granularity-outgoing edges
                    while (__any_sync(FULL_MASK, push_st<push_ed)) {
                        if (push_st < push_ed) {
                            v = csr_e[push_st];
                            testx = commi[warp_id][1] + csr_idx[push_st];
                            msg = commr[warp_id]*csr_w[testx];
                            if(v<vert_num){
                                atomicAdd(messages + v, msg);
                                isactive[v] = true;
                            }else{
                                int cur = atomicAdd(active_vert_num2,2);
                                active_vert2[cur]=v;
                                active_vert2[cur+1]=testx;
                                active_w2[cur/2] = commr[warp_id];
                            }
                        }
                        push_st += THREADS_PER_WARP; //until all outgoing edges of "u" have been processed
                    }
                } //until all source vertices with "todo_edges_num>=THREADS_PER_WARP" have been processed

                //while(3) then, the out-degree of "u" is less than THREADS_PER_WARP(32)
                int thread_count = row_end - row_start;
                int deg = thread_count;
                int scatter = 0, total = 0;

                __syncthreads();
                BlockScan(block_temp_storage).ExclusiveSum(thread_count, scatter, total); //
                __syncthreads(); //there are "total" edges left in every block
                int progress = 0;
                while (progress < total) {
                    int remain = total - progress;
                    while (scatter<(progress+THREADS_PER_BLOCK) && (row_start<row_end)) {
                        comm2[scatter-progress] = row_start;//存U有的外邻居
                        commd2[scatter-progress] = deg; //
                        commr2[scatter-progress] = ru;
                        commi2[scatter - progress][1] = idxx;
                        scatter++;
                        row_start++;
                    }
                    __syncthreads();
                    int cur_batch_count = min(remain, (int)THREADS_PER_BLOCK); //how many threads are required?
                    if (thread_id < cur_batch_count) {
                        v = csr_e[comm2[thread_id]];
                        testx = commi2[thread_id][1] + csr_idx[comm2[thread_id]];
                        msg = (commr2[thread_id]) *csr_w[testx];
                        if(v<vert_num){
                            atomicAdd(messages + v, msg);
                            isactive[v] = true;
                        }else{
                            int cur = atomicAdd(active_vert_num2,2);
                            active_vert2[cur]=v;
                            active_vert2[cur+1]=testx;
                            active_w2[cur/2] =commr2[thread_id];
                        }
                    }
                    __syncthreads();
                    progress += THREADS_PER_BLOCK;
                }
                //schedule (blockDim.x * gridDim.x) active vertices per <0>-while loop
                schedule_offset += blockDim.x * gridDim.x;
            }
            __syncthreads();



            __threadfence();
            if (threadIdx.x == 0) {
                atomicAdd((int*) &g_mutex2, 1);
                while (g_mutex2 < gridDim.x * iteration_id[l_iteration_id]) {}

            }
            __syncthreads();

            //barrier 将新的活跃顶点加入边界
            volatile __shared__ size_t output_cta_offset;
            size_t thread_idx = threadIdx.x;
            size_t schedule_offset_barrir = blockDim.x * blockIdx.x;
            while (__syncthreads_or(schedule_offset_barrir < vert_num)) {
                vid = schedule_offset_barrir + thread_idx;
                int thread_cnt = 0;
                if (vid < vert_num) {
                    if (isactive[vid]) {
                        residual[vid] += messages[vid];
                        messages[vid] = 0;
                        isactive[vid] = false;
                        if ((residual[vid]) > (csr_ov[vid+1] - csr_ov[vid])*rmax) {//执行边界检测标准，符合条件将标志位设>置为1
                            pagerank[vid] += alpha * residual[vid];
                            thread_cnt = 1;
                        }
                    }
                }
                int scatter = 0, total = 0;

                __syncthreads();
                BlockScan(block_temp_storage).ExclusiveSum(thread_cnt, scatter, total);
                __syncthreads();
                if (thread_id == 0) {
                    output_cta_offset = atomicAdd(active_vert_num, total); //run per block
                }
                __syncthreads();
                if (thread_cnt > 0) {
                    active_vert[output_cta_offset+scatter] = vid;
                }
                schedule_offset_barrir += blockDim.x * gridDim.x;
            }

            __syncthreads();
            __threadfence();
            if (thread_id == 0) {
                atomicAdd((int*) &g_mutex4, 1);
                while (g_mutex4 < gridDim.x * iteration_id[l_iteration_id]) {}

            }
            __syncthreads();

            *active_vert_num1 = 0;

            __threadfence();
            if (threadIdx.x == 0) {
                atomicAdd((int *) &g_mutex5, 1);
                while (g_mutex5 < gridDim.x * iteration_id[l_iteration_id]) {}
             //   iteration_num[0] = l_iteration_id;
              //  if(l_iteration_id<2000)	iteration_act_num[l_iteration_id]= (*active_vert_num)+(*active_vert_num2)/2;
                //printf("%d -",iteration_num[0]);
            }

            __threadfence();
        }else{
            //先顶点遍历
            __syncthreads();
            total_avtive_num=*active_vert_num;

            __syncthreads();

            size_t schedule_offset = blockDim.x * blockIdx.x;
            size_t idx=0;
            int row_start=0, row_end=0,testx=0;
            int u=0, v=0, root=0, idxx=0;
            ValueType ru=0.0, msg=0.0; //dis value of u
            while (schedule_offset < total_avtive_num) {
                idx = schedule_offset + thread_id;
                if (idx < total_avtive_num) {
                    u = active_vert[idx];
                    ru =(1-alpha)* residual[u];
                    idxx=csr_ov[u];
                    residual[u] = 0;
                    row_start = csr_v[u]; //start offset of outgoing edges of "u" in "col_ind"
                    row_end = csr_v[u+1]; //end offset of outgoing edges of "u" in "col_ind" (exclusive)
                } else {
                    row_start = 0;
                    row_end = 0;
                }
                //while(1)
                while (__syncthreads_or((row_end-row_start)>=THREADS_PER_BLOCK)) {
                    if ((row_end-row_start) >= THREADS_PER_BLOCK) {
                        comm[0][0] = thread_id; //I (thread_id) want to process the active vertex assigned to me.
                    }
                    __syncthreads(); //all threads in one block vote to processing their own vertices

                    if (comm[0][0] == thread_id) {
                        comm[0][1] = row_start; //the vertx owned by me will be processed in this <1>-while loop.
                        comm[0][2] = row_end;
                        commr[0] = ru;//ru是u的残差
                        commi[0][1] = idxx;
                        row_start = row_end; //avoid processing this vertex repeatedly in <2>&<3>-while
                    }
                    __syncthreads(); //all threads are ready to process the selected vertex

                    size_t push_st = comm[0][1] + thread_id; //process the "push_st"-th outgoing edge at first.
                    size_t push_ed = comm[0][2];

                    // <1.1>-while: block-granularity-outgoing edges
                    while (__syncthreads_or(push_st<push_ed)) {
                        if (push_st < push_ed) {
                            v = csr_e[push_st]; //target vertex id
                            testx = commi[0][1] + csr_idx[push_st];
                            msg = commr[0]*csr_w[testx];
                            if(v<vert_num){
                                atomicAdd(messages + v, msg);
                                isactive[v] = true;
                            }else{
                                int cur = atomicAdd(active_vert_num2,2);
                                active_vert2[cur]=v;
                                active_vert2[cur+1]=testx;
                                active_w2[cur/2] = commr[0];
                            }
                        }
                        push_st += THREADS_PER_BLOCK; //until all outgoing edges of "u" have been processed
                    }
                } //until all source vertices with "todo_edges_num>=THREADS_PER_BLOCK" have been processed
                //while(2)
                while (__any_sync(FULL_MASK, (row_end-row_start)>=THREADS_PER_WARP)) {
                    if ((row_end-row_start) >= THREADS_PER_WARP) {
                        comm[warp_id][0] = lane_id; //threads in the "warp_id"-th warp try to vote
                    }
                    if (comm[warp_id][0] == lane_id) {
                        comm[warp_id][1] = row_start; //vertex owned by the "lane_id"-th thread in a warp is scheduled
                        comm[warp_id][2] = row_end;
                        commr[warp_id] = ru;
                        commi[warp_id][1] = idxx;
                        row_start = row_end; //avoid processing this vertex repeatedly in <3>-while
                    }
                    size_t push_st = comm[warp_id][1] + lane_id; //process the "push_st"-th outgoing edge at first.
                    size_t push_ed = comm[warp_id][2];

                    // <2.1>-while: warp-granularity-outgoing edges
                    while (__any_sync(FULL_MASK, push_st<push_ed)) {
                        if (push_st < push_ed) {
                            v = csr_e[push_st];
                            testx = commi[warp_id][1] + csr_idx[push_st];
                            msg = commr[warp_id]*csr_w[testx];
                            if(v<vert_num){
                                atomicAdd(messages + v, msg);
                                isactive[v] = true;
                            }else{
                                int cur = atomicAdd(active_vert_num2,2);
                                active_vert2[cur]=v;
                                active_vert2[cur+1]=testx;
                                active_w2[cur/2] = commr[warp_id];
                            }
                        }
                        push_st += THREADS_PER_WARP; //until all outgoing edges of "u" have been processed
                    }
                } //until all source vertices with "todo_edges_num>=THREADS_PER_WARP" have been processed

                //while(3) then, the out-degree of "u" is less than THREADS_PER_WARP(32)
                int thread_count = row_end - row_start;
                int deg = thread_count;
                int scatter = 0, total = 0;

                __syncthreads();
                BlockScan(block_temp_storage).ExclusiveSum(thread_count, scatter, total); //
                __syncthreads(); //there are "total" edges left in every block
                int progress = 0;
                while (progress < total) {
                    int remain = total - progress;
                    while (scatter<(progress+THREADS_PER_BLOCK) && (row_start<row_end)) {
                        comm2[scatter-progress] = row_start;//存U有的外邻居
                        commd2[scatter-progress] = deg; //
                        commr2[scatter-progress] = ru;
                        commi2[scatter - progress][1] = idxx;
                        scatter++;
                        row_start++;
                    }
                    __syncthreads();
                    int cur_batch_count = min(remain, (int)THREADS_PER_BLOCK); //how many threads are required?
                    if (thread_id < cur_batch_count) {
                        v = csr_e[comm2[thread_id]];
                        testx = commi2[thread_id][1] + csr_idx[comm2[thread_id]];
                        msg = (commr2[thread_id]) *csr_w[testx];
                        if(v<vert_num){
                            atomicAdd(messages + v, msg);
                            isactive[v] = true;
                        }else{
                            int cur = atomicAdd(active_vert_num2,2);
                            active_vert2[cur]=v;
                            active_vert2[cur+1]=testx;
                            active_w2[cur/2] =commr2[thread_id];
                        }
                    }
                    __syncthreads();
                    progress += THREADS_PER_BLOCK;
                }
                //schedule (blockDim.x * gridDim.x) active vertices per <0>-while loop
                schedule_offset += blockDim.x * gridDim.x;
            }
            __syncthreads();
            __threadfence();
            if (threadIdx.x == 0) {
                atomicAdd((int *) &g_mutex1, 1);
                while (g_mutex1 < gridDim.x * iteration_id[l_iteration_id]) {}
            }
            __syncthreads();
            //host 有一个操作，将 active_verts_num 设置为0
            *active_vert_num = 0;
            total_avtive_num= *active_vert_num2/2;
            __syncthreads();
            __threadfence();


            schedule_offset = blockDim.x * blockIdx.x;
            while (schedule_offset < total_avtive_num) {
                idx = schedule_offset + thread_id;
                if (idx < total_avtive_num) {
                    u = active_vert2[idx*2];
                    idxx=active_vert2[idx*2+1];
                    ru = active_w2[idx];
                    row_start = csr_v[u]; //start offset of outgoing edges of "u" in "col_ind"
                    row_end = csr_v[u+1]; //end offset of outgoing edges of "u" in "col_ind" (exclusive)
                } else {
                    row_start = 0;
                    row_end = 0;
                }
                //while(1)
                while (__syncthreads_or((row_end-row_start)>=THREADS_PER_BLOCK)) {
                    if ((row_end-row_start) >= THREADS_PER_BLOCK) {
                        comm[0][0] = thread_id; //I (thread_id) want to process the active vertex assigned to me.
                    }
                    __syncthreads(); //all threads in one block vote to processing their own vertices

                    if (comm[0][0] == thread_id) {
                        comm[0][1] = row_start; //the vertx owned by me will be processed in this <1>-while loop.
                        comm[0][2] = row_end;
                        commr[0] = ru;//ru是u的残差
                        commi[0][1] = idxx;
                        row_start = row_end; //avoid processing this vertex repeatedly in <2>&<3>-while
                    }
                    __syncthreads(); //all threads are ready to process the selected vertex

                    size_t push_st = comm[0][1] + thread_id; //process the "push_st"-th outgoing edge at first.
                    size_t push_ed = comm[0][2];

                    // <1.1>-while: block-granularity-outgoing edges
                    while (__syncthreads_or(push_st<push_ed)) {
                        if (push_st < push_ed) {
                            v = csr_e[push_st]; //target vertex id
                            testx = commi[0][1] + csr_idx[push_st];
                            msg = commr[0]*csr_w[testx];
                            if(v<vert_num){
                                atomicAdd(messages + v, msg);
                                isactive[v] = true;
                            }else{
                                int cur = atomicAdd(active_vert_num1,2);
                                active_vert1[cur]=v;
                                active_vert1[cur+1]=testx;
                                active_w1[cur/2] = commr[0];
                            }
                        }
                        push_st += THREADS_PER_BLOCK; //until all outgoing edges of "u" have been processed
                    }
                } //until all source vertices with "todo_edges_num>=THREADS_PER_BLOCK" have been processed
                //while(2)
                while (__any_sync(FULL_MASK, (row_end-row_start)>=THREADS_PER_WARP)) {
                    if ((row_end-row_start) >= THREADS_PER_WARP) {
                        comm[warp_id][0] = lane_id; //threads in the "warp_id"-th warp try to vote
                    }
                    if (comm[warp_id][0] == lane_id) {
                        comm[warp_id][1] = row_start; //vertex owned by the "lane_id"-th thread in a warp is scheduled
                        comm[warp_id][2] = row_end;
                        commr[warp_id] = ru;
                        commi[warp_id][1] = idxx;
                        row_start = row_end; //avoid processing this vertex repeatedly in <3>-while
                    }
                    size_t push_st = comm[warp_id][1] + lane_id; //process the "push_st"-th outgoing edge at first.
                    size_t push_ed = comm[warp_id][2];

                    // <2.1>-while: warp-granularity-outgoing edges
                    while (__any_sync(FULL_MASK, push_st<push_ed)) {
                        if (push_st < push_ed) {
                            v = csr_e[push_st];
                            testx = commi[warp_id][1] + csr_idx[push_st];
                            msg = commr[warp_id]*csr_w[testx];
                            if(v<vert_num){
                                atomicAdd(messages + v, msg);
                                isactive[v] = true;
                            }else{
                                int cur = atomicAdd(active_vert_num1,2);
                                active_vert1[cur]=v;
                                active_vert1[cur+1]=testx;
                                active_w1[cur/2] = commr[warp_id];
                            }
                        }
                        push_st += THREADS_PER_WARP; //until all outgoing edges of "u" have been processed
                    }
                } //until all source vertices with "todo_edges_num>=THREADS_PER_WARP" have been processed

                //while(3) then, the out-degree of "u" is less than THREADS_PER_WARP(32)
                int thread_count = row_end - row_start;
                int deg = thread_count;
                int scatter = 0, total = 0;

                __syncthreads();
                BlockScan(block_temp_storage).ExclusiveSum(thread_count, scatter, total); //
                __syncthreads(); //there are "total" edges left in every block
                int progress = 0;
                while (progress < total) {
                    int remain = total - progress;
                    while (scatter<(progress+THREADS_PER_BLOCK) && (row_start<row_end)) {
                        comm2[scatter-progress] = row_start;//存U有的外邻居
                        commd2[scatter-progress] = deg; //
                        commr2[scatter-progress] = ru;
                        commi2[scatter - progress][1] = idxx;
                        scatter++;
                        row_start++;
                    }
                    __syncthreads();
                    int cur_batch_count = min(remain, (int)THREADS_PER_BLOCK); //how many threads are required?
                    if (thread_id < cur_batch_count) {
                        v = csr_e[comm2[thread_id]];
                        testx = commi2[thread_id][1] + csr_idx[comm2[thread_id]];
                        msg = (commr2[thread_id]) *csr_w[testx];
                        if(v<vert_num){
                            atomicAdd(messages + v, msg);
                            isactive[v] = true;
                        }else{
                            int cur = atomicAdd(active_vert_num1,2);
                            active_vert1[cur]=v;
                            active_vert1[cur+1]=testx;
                            active_w1[cur/2] =commr2[thread_id];
                        }
                    }
                    __syncthreads();
                    progress += THREADS_PER_BLOCK;
                }
                //schedule (blockDim.x * gridDim.x) active vertices per <0>-while loop
                schedule_offset += blockDim.x * gridDim.x;
            }
            __syncthreads();



            __threadfence();
            if (threadIdx.x == 0) {
                atomicAdd((int*) &g_mutex2, 1);
                while (g_mutex2 < gridDim.x * iteration_id[l_iteration_id]) {}

            }
            __syncthreads();

            //barrier 将新的活跃顶点加入边界
            volatile __shared__ size_t output_cta_offset;
            size_t thread_idx = threadIdx.x;
            size_t schedule_offset_barrir = blockDim.x * blockIdx.x;
            while (__syncthreads_or(schedule_offset_barrir < vert_num)) {
                vid = schedule_offset_barrir + thread_idx;
                int thread_cnt = 0;
                if (vid < vert_num) {
                    if (isactive[vid]) {
                        residual[vid] += messages[vid];
                        messages[vid] = 0;
                        isactive[vid] = false;
                        if ((residual[vid]) > (csr_ov[vid+1] - csr_ov[vid])*rmax) {//执行边界检测标准，符合条件将标志位设>置为1
                            pagerank[vid] += alpha * residual[vid];
                            thread_cnt = 1;
                        }
                    }
                }
                int scatter = 0, total = 0;

                __syncthreads();
                BlockScan(block_temp_storage).ExclusiveSum(thread_cnt, scatter, total);
                __syncthreads();
                if (thread_id == 0) {
                    output_cta_offset = atomicAdd(active_vert_num, total); //run per block
                }
                __syncthreads();
                if (thread_cnt > 0) {
                    active_vert[output_cta_offset+scatter] = vid;
                }
                schedule_offset_barrir += blockDim.x * gridDim.x;
            }

            __syncthreads();
            __threadfence();
            if (thread_id == 0) {
                atomicAdd((int*) &g_mutex4, 1);
                while (g_mutex4 < gridDim.x * iteration_id[l_iteration_id]) {}

            }
            __syncthreads();

            *active_vert_num2 = 0;

            __threadfence();
            if (threadIdx.x == 0) {
                atomicAdd((int *) &g_mutex5, 1);
                while (g_mutex5 < gridDim.x * iteration_id[l_iteration_id]) {}
            }

            __threadfence();
        }

        __threadfence();
        __syncthreads();
    }


}



