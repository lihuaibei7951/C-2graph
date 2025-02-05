#include <time.h>
#include "core/Util.cuh"
#include "DeviceMemory.cuh"

#include <cuda_runtime.h>
#include <sys/time.h>


__device__ volatile int g_mutex1;
__device__ volatile int g_mutex2;
__device__ volatile int g_mutex3;
__device__ volatile int g_mutex4;


template<typename ValueType>
__global__ void calcuatePPR(const int *csr_v, const int *csr_e, const ValueType *csr_w,
                            ValueType *pagerank, ValueType *residual, ValueType *messages, int *active_verts, int *active_verts_num,
                            bool *isactive, const int vert_num, const ValueType alpha, const ValueType rmax,
                            int source, int* iteration_id,int *iteration_num,int *iteration_act_num);

// Dump results
void DumpResults(const int verts_num, ValueType *d_pagerank, ValueType *d_residual, ValueType *d_messages);


int main(int argc, char **argv) {
    // Initialize graph data in host & device memory
    cudaFree(0);
    // 获取命令行参数
    std::string dir = argv[1];
    int source = atoi(argv[2]);

    // 获取，csr_v ,csr_e ,v_r,degree,order;
    Graph graph(dir);

    DeviceMemory device_memory(graph.vert_num, graph.edge_num);

    device_memory.CudaMemcpyGraph(graph);
    std::cout << "test for study how to use cuda" << endl;
    int vert_num = graph.vert_num;
    int edge_num = graph.edge_num;


    ValueType *h_pagerank = new ValueType[vert_num];
    ValueType *h_residual = new ValueType[vert_num];

    
    cout << "\n==================== PPR with FORWARD PUSH starts ====================" << endl;

    int *iteration_id;
    cudaMalloc(&iteration_id, sizeof(int)*1000);



    int *iteration_num = new int[1];
    int *iteration_act_num = new int[1000];

    CUDA_ERROR(cudaMemset(device_memory.active_verts_numStream, 0, sizeof(int)));//memset(指针， 初始值，大小）初始化

    // Initialize parameters for PPR
    
    ValueType alpha = 0.2f;
    ValueType rmax =0.001f/(graph.vert_num+1);
    int cnt = 0;

    struct timeval t_start, t_stop;
    double timeuse;
    gettimeofday(&t_start, NULL);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    while(1){
//        if(graph.csr_v[source+1]-graph.csr_v[source]<10){
//            ++source;
//            continue;
//        }
        cnt++;
        calcuatePPR<ValueType><<<MAX_BLOCKS_NUM, THREADS_PER_BLOCK>>>(
                device_memory.csr_v, device_memory.csr_e, device_memory.csr_w,device_memory.pagerankStream,
                device_memory.residualStream, device_memory.messagesStream, device_memory.active_vert,
                device_memory.active_verts_numStream, device_memory.isactive, vert_num, alpha,
                rmax, source, iteration_id,device_memory.iteration_num,device_memory.iteration_act_num);
        cudaDeviceSynchronize();
        if(cnt==10){
            //
            cudaDeviceSynchronize();
            cout << "source = " << source << "end ------------" <<cnt<< endl;
            break;
        }
    }


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float runtime = 0; //milliseconds
    cudaEventElapsedTime(&runtime, start, stop);

    cout << "gpu runtime: " << runtime/1000.0 << " seconds" << endl;
    CUDA_ERROR(cudaMemcpy(iteration_num, device_memory.iteration_num, sizeof(int)*1, cudaMemcpyDeviceToHost));
    cout << "flag 已设置成 -1  终止条件以满足		iteration_num："<<iteration_num[0]<<endl;
//    CUDA_ERROR(cudaMemcpy(iteration_act_num, device_memory.iteration_act_num, sizeof(int)*1000, cudaMemcpyDeviceToHost));
//
//    cout << "0	act_num：1"<<endl;
//    for(int i = 1 ;i<1000  ;i++){
//        if(iteration_act_num[i]==0) break;
//        cout <<i<< "	act_num："<<iteration_act_num[i]<<endl;
//
//    }
    cout << "==================== PPR with FORWARD PUSH ends ====================\n" << endl;

    //cout << "内存开辟耗时: " << timeMalloc << endl;
    gettimeofday(&t_stop, NULL);
    DumpResults(graph.vert_num, device_memory.pagerankStream, device_memory.residualStream, device_memory.messagesStream);
    timeuse = (t_stop.tv_sec - t_start.tv_sec) + (double)(t_stop.tv_usec - t_start.tv_usec)/1000000.0;
    //cout << "main total timeval runtime: " << timeuse << " seconds" << endl;
    return 0;
}


template<typename ValueType>
__global__ void calcuatePPR(const int *csr_v, const int *csr_e, const ValueType *csr_w,
                            ValueType *pagerank, ValueType *residual, ValueType *messages, int *active_verts, int *active_verts_num,
                            bool *isactive, const int vert_num, const ValueType alpha, const ValueType rmax,
                            int source, int* iteration_id,int *iteration_num,int *iteration_act_num) {
    size_t thread_id = threadIdx.x;
    size_t schedule_offset_init = blockDim.x * blockIdx.x;
    size_t vid = 0;
    while (schedule_offset_init < vert_num) {
        vid = schedule_offset_init + thread_id;
        //in the last batch, some threads are idle
        if (vid < vert_num) {
            pagerank[vid] = 0;
            residual[vid] = 0;
            isactive[vid] = false;
            messages[vid] = 0;
        }
        if (vid < 1000) {
            iteration_id[vid] = vid;
        }
        schedule_offset_init += blockDim.x * gridDim.x;
    }

    //prepare for the 1st iteration
    size_t global_id = thread_id + blockDim.x*blockIdx.x;
    if (global_id == 0) { //每一个块中线程为0的id，source=1,第一个顶点的residual值初始化为1，
        residual[source] = 1;
        *active_verts_num = 1;
        active_verts[0] = source;//当前迭代中的活跃顶点
        pagerank[source] += alpha * residual[source];
        g_mutex1 = 0;
        g_mutex2 = 0;
        g_mutex4 = 0;

        //g_mutex3 = 0;
        //printf("source = %d 初始化完成\n", source);
    }
    ////同步点

    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        atomicAdd((int*) &g_mutex3, 1);
        while ((g_mutex3 == 0) || (g_mutex3 % gridDim.x) ) {}
        //	printf("g_mutex1 = %d \t阈值 : %d\n", g_mutex1, gridDim.x * iteration_id[l_iteration_id]);
    }
    __syncthreads();


    int l_iteration_id = 0;
    int l_active_verts_num = *active_verts_num;
    //size_t thread_id = threadIdx.x;
    size_t lane_id = thread_id % THREADS_PER_WARP;
    size_t warp_id = thread_id / THREADS_PER_WARP; //the i-th warp (from 0)  当前块内warp的id
    typedef cub::BlockScan<int, THREADS_PER_BLOCK> BlockScan;
    __shared__ typename BlockScan::TempStorage block_temp_storage;
    volatile __shared__ int comm[THREADS_PER_BLOCK/THREADS_PER_WARP][3];//[256/32][3]第一维是多少个warp，256/32=8，8个warp，
    volatile __shared__ ValueType commr[THREADS_PER_BLOCK/THREADS_PER_WARP];//每个warp对应一个值
    volatile __shared__ int comm2[THREADS_PER_BLOCK]; //一维数组大小256，int
    volatile __shared__ ValueType commd2[THREADS_PER_BLOCK]; //out-degree
    volatile __shared__ ValueType commr2[THREADS_PER_BLOCK];


    while (*active_verts_num > 0&&l_iteration_id<500) {



        l_iteration_id += 1;
        //pushmessages 当前活跃顶点发消息
        ////同步点
        __syncthreads();
        int total_active_verts_num = *active_verts_num;
        __syncthreads();

        __threadfence();
        if (thread_id == 0) {
            atomicAdd((int*) &g_mutex4, 1);
            while (g_mutex4 < gridDim.x * iteration_id[l_iteration_id]) {}

        }
        __syncthreads();
        size_t schedule_offset = blockDim.x * blockIdx.x;
        size_t idx = 0;
        int u = 0;
        int row_start, row_end;
        int v;
        ValueType ru, msg;
        while (schedule_offset < total_active_verts_num) {
            idx = schedule_offset + thread_id;
            if (idx < total_active_verts_num) {
                u = active_verts[idx];
                ru = residual[u];
                residual[u] = 0;
                row_start = csr_v[u]; //start offset of outgoing edges of "u" in "col_ind"
                row_end = csr_v[u+1]; //end offset of outgoing edges of "u" in "col_ind" (exclusive)
                //补全0出度点的发送方向为源顶点
                /*if(csr_v[u]==csr_v[u + 1]) {
                    row_start = csr_v[source];
                    row_end = csr_v[source + 1];

                }*/
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
                    row_start = row_end; //avoid processing this vertex repeatedly in <2>&<3>-while
                }
                __syncthreads(); //all threads are ready to process the selected vertex

                size_t push_st = comm[0][1] + thread_id; //process the "push_st"-th outgoing edge at first.
                size_t push_ed = comm[0][2];

                // <1.1>-while: block-granularity-outgoing edges
                while (__syncthreads_or(push_st<push_ed)) {
                    if (push_st < push_ed) {
                        v = csr_e[push_st]; //target vertex id
                        msg = (1-alpha)*commr[0]*csr_w[push_st];
                        //msg = ((1-alpha)*commr[0]) / (comm[0][2]-comm[0][1]);

                        atomicAdd(messages + v, msg);
                        isactive[v] = true;//目标顶点设置成活跃
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
                    row_start = row_end; //avoid processing this vertex repeatedly in <3>-while
                }
                size_t push_st = comm[warp_id][1] + lane_id; //process the "push_st"-th outgoing edge at first.
                size_t push_ed = comm[warp_id][2];

                // <2.1>-while: warp-granularity-outgoing edges
                while (__any_sync(FULL_MASK, push_st<push_ed)) {
                    if (push_st < push_ed) {
                        v = csr_e[push_st];
                        msg = ((1-alpha)*commr[warp_id]) *csr_w[push_st];
                        //msg = ((1-alpha)*commr[warp_id]) / (comm[warp_id][2]-comm[warp_id][1]);

                        atomicAdd(messages + v, msg);
                        isactive[v] = true;
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
                    scatter++;
                    row_start++;
                }
                __syncthreads();
                int cur_batch_count = min(remain, (int)THREADS_PER_BLOCK); //how many threads are required?
                if (thread_id < cur_batch_count) {
                    v = csr_e[comm2[thread_id]];
                    msg = ((1-alpha)*commr2[thread_id]) *csr_w[comm2[thread_id]];
                    //msg = ((1-alpha)*commr2[thread_id]) / commd2[thread_id];

                    atomicAdd(messages + v, msg);
                    isactive[v] = true;
                }
                __syncthreads();
                progress += THREADS_PER_BLOCK;
            }
            //schedule (blockDim.x * gridDim.x) active vertices per <0>-while loop
            schedule_offset += blockDim.x * gridDim.x;
        }
        __syncthreads();
        //host 有一个操作，将 active_verts_num 设置为0
        *active_verts_num = 0;

        __threadfence();
        if (threadIdx.x == 0) {
            atomicAdd((int*) &g_mutex1, 1);
            while (g_mutex1 < gridDim.x * iteration_id[l_iteration_id]) {}

        }
        __syncthreads();

        //barrier 将新的活跃顶点加入边界
        volatile __shared__ size_t output_cta_offset;
        size_t thread_idx = threadIdx.x;
        size_t schedule_offset_barrir = blockDim.x * blockIdx.x;
        size_t vid = 0;

        while (__syncthreads_or(schedule_offset_barrir < vert_num)) {
            vid = schedule_offset_barrir + thread_idx;
            int thread_cnt = 0;
            if (vid < vert_num) {
                if (isactive[vid]) {
                    residual[vid] += messages[vid];
                    messages[vid] = 0;
                    isactive[vid] = false;
                    if (residual[vid] > rmax) {//执行边界检测标准，符合条件将标志位设>置为1
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
                output_cta_offset = atomicAdd(active_verts_num, total); //run per block
            }
            __syncthreads();
            if (thread_cnt > 0) {
                active_verts[output_cta_offset+scatter] = vid;
                //if (l_iteration_id == 45) printf("%d\t", vid);
            }
            schedule_offset_barrir += blockDim.x * gridDim.x;
        }

        __syncthreads();
        __threadfence();
        if (thread_id == 0) {
            atomicAdd((int*) &g_mutex2, 1);
            while (g_mutex2 < gridDim.x * iteration_id[l_iteration_id]) {}

        }
        __syncthreads();
        __threadfence();
        if (thread_id == 0) {
            iteration_num[0] = l_iteration_id;
            if(l_iteration_id<1000)	iteration_act_num[l_iteration_id]= *active_verts_num;

        }
        __syncthreads();

        //l_active_verts_num = *active_verts_num;

    } //while (*active_verts_num != 0);
}

//Dump result
void DumpResults(const int verts_num, ValueType *d_pagerank, ValueType *d_residual, ValueType *d_messages) {
    ValueType *h_pagerank = new ValueType[verts_num];
    ValueType *h_residual = new ValueType[verts_num];
    ValueType *h_messages = new ValueType[verts_num];

    CUDA_ERROR(cudaMemcpy(h_pagerank, d_pagerank,
                          verts_num*sizeof(ValueType), cudaMemcpyDeviceToHost));

    CUDA_ERROR(cudaMemcpy(h_residual, d_residual,
                          verts_num*sizeof(ValueType), cudaMemcpyDeviceToHost));

    CUDA_ERROR(cudaMemcpy(h_messages, d_messages,
                          verts_num*sizeof(ValueType), cudaMemcpyDeviceToHost));

 //   ofstream outfile("out/iter_2.txt");

    for (int i = 0; i <=10; i++) {
        cout<<i<<".\tpagerank\t "<<h_pagerank[i] << "\tresidual\t" <<h_residual[i] <<endl;
    }

//    for (int i = 0; i < 100000; i++) {
//        outfile<<i;
//        outfile<<" ";
//        outfile<<h_pagerank[i];
//  //      outfile<<" ";
//   //     outfile<<h_residual[i];
//        outfile<<"\n";
//    }

    delete[] h_residual;
    delete[] h_pagerank;
    delete[] h_messages;
    h_residual = NULL;
    h_pagerank = NULL;
    h_messages = NULL;
}

