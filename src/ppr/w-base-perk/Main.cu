#include <time.h>
#include "core/Util.cuh"
#include "DeviceMemory.cuh"
#include "PPR.h"
#include <sys/time.h>

//#include <cuda_runtime.h>
using namespace std;
__device__ volatile int g_mutex1;
__device__ volatile int g_mutex2;
__device__ volatile int g_mutex3;
__device__ volatile int g_mutex4; // rule 遍历
__device__ volatile int g_mutex5; // rule 消息传播
__device__ volatile int g_mutex6; // rule 消息传播
__device__ volatile int g_mutex7; // rule 消息传播


template <typename ValueType>
__global__ void
calcuatePPR(const int *csr_v, const int *csr_e, const ValueType *csr_w,ValueType *pagerank,
            ValueType *residual, ValueType *messages, int *active_vert,
            int *active_verts_num, bool *isactive, const int vert_num,
            const ValueType alpha, const ValueType rmax,
            int source, int *iteration_id,int iter);

// Dump results
void DumpResults(const int verts_num, ValueType *d_pagerank, ValueType *d_residual, ValueType *d_messages);

int main(int argc, char **argv) {
    // Initialize graph data in host & device memory
    cudaFree(0);



    std::string dir = argv[1];
    int source = atoi(argv[2]);
    int iter = atoi(argv[3]);
    int iterx=iter;
    graph.Graphinit(dir);

    DeviceMemory device_memory(graph.vert_num,  graph.edge_num);

    device_memory.CudaMemcpyGraph(graph);
    std::cout << "test for study how to use cuda" << endl;

    int *iteration_id;
    cudaMalloc(&iteration_id, sizeof(int) * 2000);


    // Initialize parameters for PPR


    //init bufferqueue
    BufferQueue *bufferqueue;
    bufferqueue = new BufferQueue;
    initQueue(bufferqueue, 20);

    //create thread
    pthread_t Pthread;

    if(0 != pthread_create(&Pthread, NULL, ppr_CPU, (void*)bufferqueue)){
        printf("Error:线程创建失败\n");
        exit(-1);
    }

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);


    cout << "\n==================== PPR with FORWARD PUSH starts ====================" << endl;


    // Initialize parameters for PPR

    int *flagG;
    CUDA_ERROR(cudaMalloc(&flagG, sizeof(int)));
    CUDA_ERROR(cudaMemset(flagG, -1, sizeof(int)));

    struct timeval t_start, t_stop;
    double timeuse;
    gettimeofday(&t_start, NULL);

    int cnt = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    while (1) {

        //cout << "source = " << source << "start ------------" << endl;
        while(graph.csr_v[source+1]-graph.csr_v[source]<10){
            ++source;
        }

        calcuatePPR<ValueType><<<MAX_BLOCKS_NUM, THREADS_PER_BLOCK, 0, stream2>>>(
                device_memory.csr_v, device_memory.csr_e,device_memory.csr_w,
                device_memory.pagerankStream2, device_memory.residualStream2,
                device_memory.messagesStream, device_memory.active_vert2,
                device_memory.active_verts_numStream2,
                device_memory.isactive, graph.vert_num,
                alpha, graph.rmax, source++, iteration_id,iter);
        if(cnt){
            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->global_ft_cnt, device_memory.active_verts_numStream1,
                                       sizeof(int), cudaMemcpyDeviceToHost, stream1));

            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->reserve, device_memory.pagerankStream1,
                                       graph.vert_num*sizeof(ValueType), cudaMemcpyDeviceToHost, stream1));

            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->residue, device_memory.residualStream1,
                                       graph.vert_num*sizeof(ValueType), cudaMemcpyDeviceToHost, stream1));

            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->global_ft, device_memory.active_vert1,
                                       graph.vert_num *sizeof(int), cudaMemcpyDeviceToHost, stream1));

            //这个是关键同步机制？？？
            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->flag, flagG,
                                       sizeof(int), cudaMemcpyDeviceToHost, stream1));


            if(bufferqueue->length>4){
                if(bufferqueue->length<7){
                    bufferqueue->front->source = source-1;
                    bufferqueue->front = bufferqueue->front->next; // 指针后移
                    bufferqueue->length++;
                }else{
                    iter=10000;
                }

            }else{
                bufferqueue->front->source = source-1;
                bufferqueue->front = bufferqueue->front->next; // 指针后移
                bufferqueue->length++;
                iter=iterx;
            }
            cnt++;
        }

        while(graph.csr_v[source+1]-graph.csr_v[source]<10){
            ++source;
        }
        calcuatePPR<ValueType><<<MAX_BLOCKS_NUM, THREADS_PER_BLOCK, 0, stream1>>>(
                device_memory.csr_v, device_memory.csr_e,device_memory.csr_w,
                device_memory.pagerankStream1, device_memory.residualStream1,
                device_memory.messagesStream, device_memory.active_vert1,
                device_memory.active_verts_numStream1,
                device_memory.isactive, graph.vert_num,
                alpha, graph.rmax, source++, iteration_id,iter);


        CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->global_ft_cnt, device_memory.active_verts_numStream2,
                                   sizeof(int), cudaMemcpyDeviceToHost, stream2));

        CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->reserve, device_memory.pagerankStream2,
                                   graph.vert_num*sizeof(ValueType), cudaMemcpyDeviceToHost, stream2));

        CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->residue, device_memory.residualStream2,
                                   graph.vert_num*sizeof(ValueType), cudaMemcpyDeviceToHost, stream2));

        CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->global_ft, device_memory.active_vert2,
                                   graph.vert_num * sizeof(int), cudaMemcpyDeviceToHost, stream2));

        CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->flag, flagG,
                                   sizeof(int), cudaMemcpyDeviceToHost, stream2));

        // Stream2 完成后，更新队列
        cudaStreamSynchronize(stream2);
        cout << "当前bufferqueue长度为:\t" << bufferqueue->length << endl;


        if(bufferqueue->length>4){
            if(bufferqueue->length<7){
                bufferqueue->front->source = source-1;
                bufferqueue->front = bufferqueue->front->next; // 指针后移
                bufferqueue->length++;
                iter=10000;
            }else{
                iter=10000;
            }

        }else{
            bufferqueue->front->source = source-1;
            bufferqueue->front = bufferqueue->front->next; // 指针后移
            bufferqueue->length++;
            iter=iterx;
        }
        cnt++;
        cout << "当前bufferqueue长度为:\t" << bufferqueue->length << endl;

        if (cnt>=5) {
            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->global_ft_cnt, device_memory.active_verts_numStream1,
                                       sizeof(int), cudaMemcpyDeviceToHost, stream1));

            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->reserve, device_memory.pagerankStream1,
                                       graph.vert_num*sizeof(ValueType), cudaMemcpyDeviceToHost, stream1));

            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->residue, device_memory.residualStream1,
                                       graph.vert_num*sizeof(ValueType), cudaMemcpyDeviceToHost, stream1));

            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->global_ft, device_memory.active_vert1,
                                       graph.vert_num *sizeof(int), cudaMemcpyDeviceToHost, stream1));

            //这个是关键同步机制？？？
            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->flag, flagG,
                                       sizeof(int), cudaMemcpyDeviceToHost, stream1));


            cudaStreamSynchronize(stream1);
            cudaDeviceSynchronize();
            bufferqueue->front->source = source-1;
            bufferqueue->front = bufferqueue->front->next; // 指针后移
            bufferqueue->length++;
            cnt++;
            bufferqueue->flag = -1;
            cout << "缓冲队列已空  图计算完成" << endl;

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            break;


        }
    }

    cout << "==================== PPR with FORWARD PUSH ends ====================\n" << endl;

    pthread_join(Pthread, NULL);
//CPU执行之后的迭代过程
    float runtime = 0; //milliseconds
    cudaEventElapsedTime(&runtime, start, stop);
    cout << "gpu runtime: " << runtime/1000.0 << " seconds" << endl;

    gettimeofday(&t_stop, NULL);
    timeuse = (t_stop.tv_sec - t_start.tv_sec) + (double)(t_stop.tv_usec - t_start.tv_usec)/1000000.0;
    cout << "main total timeval runtime: " << timeuse << " seconds" << endl;



    return 0;
}


template <typename ValueType>
__global__ void
calcuatePPR(const int *csr_v, const int *csr_e, const ValueType *csr_w,ValueType *pagerank,
            ValueType *residual, ValueType *messages, int *active_vert,
            int *active_verts_num, bool *isactive, const int vert_num,
            const ValueType alpha, const ValueType rmax,
            int source, int *iteration_id,int iter) {

    size_t thread_id = threadIdx.x;
    size_t schedule_offset_init = blockDim.x * blockIdx.x;
    size_t vid = 0;
    while (schedule_offset_init < vert_num ) {
        vid = schedule_offset_init + thread_id;
        // in the last batch, some threads are idle
        if (vid < vert_num ) {
            pagerank[vid] = 0;
            residual[vid] = 0;
            messages[vid] = 0;
            isactive[vid] = false;
        }



        if (vid < 1000) {
            iteration_id[vid] = vid;
        }

        schedule_offset_init +=
                blockDim.x * gridDim.x; // 块线程数目*网格块数=网格线程数目
    }
    __syncthreads(); // 确保第一个初始化过程内的线程块同步


    // prepare for the 1st iteration
    size_t global_id = thread_id + blockDim.x * blockIdx.x;
    if (global_id ==
        0) { // 每一个块中线程为0的id，source=1,第一个顶点的residual值初始化为1，
        residual[source] = 1;
        *active_verts_num = 1;
        active_vert[0] = source; // 当前迭代中的活跃顶点
        pagerank[source] += alpha * residual[source];
        g_mutex1 = 0;
        g_mutex2 = 0;
        g_mutex4 = 0;
        g_mutex5 = 0;
        g_mutex6 = 0;
        // g_mutex3 = 0;
        //printf("source = %d 初始化完成\n", source);
    }

    __syncthreads(); //__syncthreads()只能在一个线程块内使用，不能用于不同线程块之间的同步。
    __threadfence(); // 线程屏障函数，它确保所有线程都在此之前的所有内存操作都已经完成。
    if (threadIdx.x == 0) {
        atomicAdd((int *)&g_mutex3, 1);
        while ((g_mutex3 == 0) || (g_mutex3 % gridDim.x)) {
        } // 用于等待其他线程块的同步的循环。
    }
    __syncthreads();

    int l_iteration_id = 0;
    int l_active_verts_num = *active_verts_num;

    size_t lane_id = thread_id % THREADS_PER_WARP; // warp内线程的id
    size_t warp_id =
            thread_id / THREADS_PER_WARP; // the i-th warp (from 0) 当前块内warp的id

    typedef cub::BlockScan<int, THREADS_PER_BLOCK> BlockScan;

    __shared__ typename BlockScan::TempStorage block_temp_storage;

    volatile __shared__ int comm[THREADS_PER_BLOCK / THREADS_PER_WARP][3];
    //[256/32][3]第一维是多少个warp，256/32=8，8个warp，

    volatile __shared__ ValueType
            commr[THREADS_PER_BLOCK / THREADS_PER_WARP]; // 每个warp对应一个值

    volatile __shared__ ValueType
            commd[THREADS_PER_BLOCK / THREADS_PER_WARP]; // 每个warp对应一个值

    volatile __shared__ int comm2[THREADS_PER_BLOCK]; // 一维数组大小256，int

    volatile __shared__ ValueType commd2[THREADS_PER_BLOCK]; // out-degree

    volatile __shared__ ValueType commr2[THREADS_PER_BLOCK];
    while ((*active_verts_num > 0) ){  //     l_iteration_id<29&&
        //       if(iter!=10000&&(*active_verts_num<500&&l_iteration_id>10)) return ;
        //&&((l_iteration_id<iter)||(*active_verts_num>10000))
        l_iteration_id += 1;
        // pushmessages 当前活跃顶点发消息

        __syncthreads();
        int total_active_verts_num = *active_verts_num;
        __threadfence();
        if (thread_id == 0) {
            atomicAdd((int *)&g_mutex5, 1);
            while (g_mutex5 < gridDim.x * iteration_id[l_iteration_id]) {
            }
        }
        __syncthreads();

        size_t idx = 0;
        int u = 0;
        int e_start, e_end;
        int v;
        ValueType ru, msg;

        // 顶点遍历
        size_t schedule_1 = blockDim.x * blockIdx.x; // 块索引
        while (schedule_1 < total_active_verts_num) {
            idx = schedule_1 + thread_id;
            if (idx < total_active_verts_num) {
                u = active_vert[idx];
                ru = residual[u];
                residual[u] = 0;
                e_start = csr_v[u]; // start offset of outgoing edges of "u"
                // in "col_ind"
                e_end = csr_v[u + 1]; // end offset of outgoing edges of "u"
                // in "col_ind" (exclusive)
            } else {
                e_start = 0;
                e_end = 0;
            }
            // while(1)
            while (
                    __syncthreads_or((e_end - e_start) >= THREADS_PER_BLOCK)) {
                if ((e_end - e_start) >= THREADS_PER_BLOCK) {
                    comm[0][0] =
                            thread_id; // I (thread_id) want to process the
                    // active vertex assigned to me.
                }
                __syncthreads(); // all threads in one block vote to
                // processing their own vertices

                if (comm[0][0] == thread_id) {
                    comm[0][1] =
                            e_start; // the vertx owned by me will be
                    // processed in this <1>-while loop.
                    comm[0][2] = e_end;
                    commr[0] = ru; // ru是u的残差

                    e_start = e_end; // avoid processing this vertex
                    // repeatedly in <2>&<3>-while
                }
                __syncthreads(); // all threads are ready to process the
                // selected vertex

                size_t push_st =
                        comm[0][1] + thread_id; // process the "push_st"-th
                // outgoing edge at first.
                size_t push_ed = comm[0][2];

                // <1.1>-while: block-granularity-outgoing edges
                while (__syncthreads_or(push_st < push_ed)) {
                    if (push_st < push_ed) {
                        v = csr_e[push_st]; // target vertex id

                        msg = ((1 - alpha) * commr[0]*csr_w[push_st]);// outdeg of the selected s, not "u"

                        atomicAdd(messages + v, msg);

                        isactive[v] =true;


                    }
                    push_st +=
                            THREADS_PER_BLOCK; // until all outgoing edges of
                    // "u" have been processed
                }
            } // until all source vertices with
            // "todo_edges_num>=THREADS_PER_BLOCK" have been processed

            // while(2)
            while (__any_sync(FULL_MASK,
                              (e_end - e_start) >= THREADS_PER_WARP)) {
                if ((e_end - e_start) >= THREADS_PER_WARP) {
                    comm[warp_id][0] =
                            lane_id; // threads in the "warp_id"-th warp try to
                    // vote
                }
                if (comm[warp_id][0] == lane_id) {
                    comm[warp_id][1] =
                            e_start; // vertex owned by the "lane_id"-th thread
                    // in a warp is scheduled
                    comm[warp_id][2] = e_end;
                    commr[warp_id] = ru;
                    e_start = e_end; // avoid processing this vertex
                    // repeatedly in <3>-while
                }
                size_t push_st =
                        comm[warp_id][1] + lane_id; // process the "push_st"-th
                // outgoing edge at first.
                size_t push_ed = comm[warp_id][2];

                // <2.1>-while: warp-granularity-outgoing edges
                while (__any_sync(FULL_MASK, push_st < push_ed)) {
                    if (push_st < push_ed) {
                        v = csr_e[push_st];
                        msg = ((1 - alpha) * commr[warp_id]*csr_w[push_st]);
                        atomicAdd(messages + v, msg);
                        isactive[v] =true;
                    }
                    push_st +=
                            THREADS_PER_WARP; // until all outgoing edges of
                    // "u" have been processed
                }
            } // until all source vertices with
            // "todo_edges_num>=THREADS_PER_WARP" have been processed

            // while(3) then, the out-degree of "u" is less than
            // THREADS_PER_WARP(32)
            int thread_count = e_end - e_start;
            int deg = thread_count;
            int scatter = 0, total = 0;

            __syncthreads();
            BlockScan(block_temp_storage)
                    .ExclusiveSum(thread_count, scatter, total); //
            __syncthreads(); // there are "total" edges left in every block
            int progress = 0;

            while (progress < total) {
                int remain = total - progress;
                while (scatter < (progress + THREADS_PER_BLOCK) &&
                       (e_start < e_end)) {
                    comm2[scatter - progress] = e_start; // 存U有的外邻居
                    commd2[scatter - progress] = deg;     //
                    commr2[scatter - progress] = ru;
                    scatter++;
                    e_start++;
                }
                __syncthreads();
                int cur_batch_count = min(remain, (int)THREADS_PER_BLOCK);
                // how many threads are required?
                if (thread_id < cur_batch_count) {

                    v = csr_e[comm2[thread_id]]; //!!!
                    //printf("the vert :%d \n",v);
                    msg = ((1 - alpha) * commr2[thread_id] * csr_w[comm2[thread_id]] ) ;
                    atomicAdd(messages + v, msg);
                    isactive[v] =true;
                }
                __syncthreads();
                progress += THREADS_PER_BLOCK;
            }
            // schedule (blockDim.x * gridDim.x) active vertices per
            // <0>-while loop
            schedule_1 += blockDim.x * gridDim.x;
        }
        //iter[0] = 0;
        __syncthreads();
        __threadfence();
        if (thread_id == 0) {
            atomicAdd((int *)&g_mutex4, 1);
            while (g_mutex4 < gridDim.x * iteration_id[l_iteration_id]) {
            }
        }
        __syncthreads();


        // 边界检测

        //host 有一个操作，将 active_verts_num 设置为0
        *active_verts_num = 0;

        __threadfence();
        if (threadIdx.x == 0) {
            atomicAdd((int*) &g_mutex1, 1);
            while (g_mutex1 < gridDim.x * iteration_id[l_iteration_id]) {}

        }
        __syncthreads();


        // barrier 将新的活跃顶点加入边界
        volatile __shared__ size_t output_cta_offset;
        size_t thread_idx = threadIdx.x;
        schedule_1 = blockDim.x * blockIdx.x;
        size_t vid = 0;

        while (__syncthreads_or(schedule_1 < vert_num)) {
            // 这个函数返回的是一个布尔值，表示所有线程块中至少有一个线程满足条件。
            vid = schedule_1 + thread_idx;
            int thread_cnt = 0;
            if (vid < vert_num) {
                if (isactive[vid]) {
                    residual[vid] += messages[vid];
                    messages[vid] = 0;
                    isactive[vid] = false;
                    if (residual[vid]  > rmax) {
                        // 执行边界检测标准，符合条件将标志位设>置为1
                        pagerank[vid] += alpha * residual[vid];
                        thread_cnt = 1;
                    }
                }
            }
            int scatter = 0, total = 0;

            __syncthreads();
            BlockScan(block_temp_storage)
                    .ExclusiveSum(thread_cnt, scatter, total);
            __syncthreads();
            if (thread_id == 0) {
                output_cta_offset =
                        atomicAdd(active_verts_num, total); // run per block
            }
            __syncthreads();
            if (thread_cnt > 0) {
                active_vert[output_cta_offset + scatter] = vid;
            }
            schedule_1 += blockDim.x * gridDim.x;
            // 用于更新 schedule_offset_barrir
            // 变量的值，并且在每个线程块内都会执行。
        } // 边界检测结束

        __syncthreads();
        __threadfence();
        if (thread_id == 0) {
            atomicAdd((int *)&g_mutex2, 1);
            while (g_mutex2 < gridDim.x * iteration_id[l_iteration_id]) {
            }
        }
        __syncthreads();
        l_active_verts_num = *active_verts_num;
        __syncthreads();
        __threadfence();

//        break;
    } // while (*active_verts_num != 0);
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

    ofstream outfile("/home/lhb/cucode/PPR/Judge/iter_2.txt");

    for (int i = 0; i <=10; i++) {
        cout<<i<<".\tpageran\t "<<h_pagerank[i] << "\tresidual\t" <<h_residual[i] <<endl;
    }

    for (int i = 0; i < verts_num; i++) {
        outfile<<i;
        outfile<<" ";
        outfile<<h_pagerank[i];
        outfile<<" ";
        outfile<<h_residual[i];
        outfile<<"\n";
    }

    delete[] h_residual;
    delete[] h_pagerank;
    delete[] h_messages;
    h_residual = NULL;
    h_pagerank = NULL;
    h_messages = NULL;
}

