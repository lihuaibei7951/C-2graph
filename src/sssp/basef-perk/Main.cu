#include <time.h>
#include "Util.cuh"
#include "DeviceMemory.cuh"
#include "SSSP.h"
#include <sys/time.h>

__device__ volatile int g_mutex1;
__device__ volatile int g_mutex2;
__device__ volatile int g_mutex3;
__device__ volatile int g_mutex4;
__device__ volatile int g_mutex5;


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

//kernel_1 init
// 检查 CUDA 错误的宏
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << "@" << __LINE__      \
                      << ": " << cudaGetErrorString(err) << std::endl;        \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)
//kernel_ update
//template<typename ValueType>
__global__ void CalcuSSSP(const Vertex *csr_v,const Vertex *csr_e, ValueType *csr_w, ValueType *dis,
                          Vertex *active_vert, Vertex *active_vert_num, bool *isactive,
                          const Vertex vert_num, Vertex source,Vertex *iteration_id,int iter);

int main(int argc, char **argv) {
    // Initialize graph data in host & device memory
    cudaFree(0);
    // 获取命令行参数
    std::string dir = argv[1];
    int source = atoi(argv[2]);
    int iter = atoi(argv[3]);
    graph.Graphinit(dir);

    DeviceMemory device_memory(graph.vert_num, graph.edge_num);

    device_memory.CudaMemcpyGraph(graph);
    std::cout << "test for study how to use cuda" << endl;

    int *iteration_id;
    cudaMalloc(&iteration_id, sizeof(int)*2000);


    BufferQueue *bufferqueue;
    bufferqueue = new BufferQueue;
    initQueue(bufferqueue, 20);//1000


    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    pthread_t Pthread;

    if(0 != pthread_create(&Pthread, NULL, SSSP_CPU, (void *)bufferqueue)){
        printf("Error: 线程创建失败\n");
        exit(-1);
    }


    cout << "\n==================== SSSP with FORWARD PUSH starts ============" <<endl;

    int cnt = 0;

    int *flagG;
    CUDA_ERROR(cudaMalloc(&flagG, sizeof(int)));
    CUDA_ERROR(cudaMemset(flagG, -1, sizeof(int)));

    struct timeval t_start, t_stop;
    double timeuse;
    gettimeofday(&t_start, NULL);


    using namespace std::chrono;

    // 存储时间点
    std::vector<high_resolution_clock::time_point> timestamps;

    // 记录第一个时间点
    timestamps.push_back(high_resolution_clock::now());
        cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);



    
    while (1) {
    while(graph.csr_v[source+1]-graph.csr_v[source]<1){//
            ++source;
        }

        CalcuSSSP<<<MAX_BLOCKS_NUM, THREADS_PER_BLOCK,0,stream2>>>(
                device_memory.csr_v, device_memory.csr_e, device_memory.csr_w, device_memory.distance2,
                device_memory.active_vert2,device_memory.active_vert_num2,device_memory.isactive,
                graph.vert_num, source++,iteration_id,iter);
        
        
        if(cnt){
       
         
            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->distance, device_memory.distance1,
                                       graph.vert_num *sizeof(ValueType), cudaMemcpyDeviceToHost, stream1 ));//1

            //额外开销

            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->global_ft_cnt, device_memory.active_vert_num1,
                                       sizeof(int), cudaMemcpyDeviceToHost, stream1));

            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->global_ft, device_memory.active_vert1,
                                       graph.vert_num *sizeof(int), cudaMemcpyDeviceToHost, stream1 ));

            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->flag, flagG,
                                       sizeof(int), cudaMemcpyDeviceToHost, stream1));

            ////更新队列信息
            cudaStreamSynchronize(stream1);
            bufferqueue->front->source = source-1;
            bufferqueue->front = bufferqueue->front->next; // 指针后移
            bufferqueue->length++;
            cnt++;
        }

while(graph.csr_v[source+1]-graph.csr_v[source]<1){//
            ++source;
        }
        CalcuSSSP<<<MAX_BLOCKS_NUM, THREADS_PER_BLOCK,0,stream1>>>(
                device_memory.csr_v, device_memory.csr_e, device_memory.csr_w, device_memory.distance1,
                device_memory.active_vert1,device_memory.active_vert_num1,device_memory.isactive,
                graph.vert_num, source++,iteration_id,iter);

        CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->distance, device_memory.distance2,
                                   graph.vert_num*sizeof(ValueType), cudaMemcpyDeviceToHost, stream2));

        //额外开销

        CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->global_ft_cnt, device_memory.active_vert_num2,
                                   sizeof(int), cudaMemcpyDeviceToHost, stream2));

        CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->global_ft, device_memory.active_vert2,
                                   graph.vert_num *sizeof(int), cudaMemcpyDeviceToHost, stream2 ));

        CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->flag, flagG,
                                   sizeof(int), cudaMemcpyDeviceToHost, stream2));

        ////更新队列信息
        // Stream2 完成后，更新队列
        cudaStreamSynchronize(stream2);
        bufferqueue->front->source = source-1;
        bufferqueue->front = bufferqueue->front->next; // 指针后移
        bufferqueue->length++;
        cnt++;
          timestamps.push_back(high_resolution_clock::now());
         cout << "当前bufferqueue长度为:\t" << bufferqueue->length << endl;


        if (cnt >= 2) {
                  
         
            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->distance, device_memory.distance1,
                                       graph.vert_num *sizeof(ValueType), cudaMemcpyDeviceToHost, stream1 ));//1

            //额外开销

            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->global_ft_cnt, device_memory.active_vert_num1,
                                       sizeof(int), cudaMemcpyDeviceToHost, stream1));

            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->global_ft, device_memory.active_vert1,
                                       graph.vert_num *sizeof(int), cudaMemcpyDeviceToHost, stream1 ));

            CUDA_ERROR(cudaMemcpyAsync(bufferqueue->front->flag, flagG,
                                       sizeof(int), cudaMemcpyDeviceToHost, stream1));

            ////更新队列信息
            cudaStreamSynchronize(stream1);
            bufferqueue->front->source = source-2;
            bufferqueue->front = bufferqueue->front->next; // 指针后移
            bufferqueue->length++;
            cnt++;
            cudaDeviceSynchronize();
            bufferqueue->flag = -1;
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            break;
        }

    }


    
    float runtime = 0;
    cudaEventElapsedTime(&runtime, start, stop);
    

    pthread_join(Pthread, NULL);
    cout << "gpu runtime: " << runtime/1000.0<< " seconds" <<endl;
    cout << "源顶点source = " << source <<endl;

    gettimeofday(&t_stop, NULL);
    timeuse = (t_stop.tv_sec - t_start.tv_sec) + (double)(t_stop.tv_usec - t_start.tv_usec)/1000000.0;
    cout << "main total timeval runtime: " << timeuse << " seconds" << endl;

 // 计算并输出时间差
    for (size_t i = 1; i < timestamps.size(); ++i) {
        auto duration = duration_cast<milliseconds>(timestamps[i] - timestamps[0]);
        std::cout << duration.count()  << std::endl;
    }

    return 0;
}


//kernel_2 update
//template<typename ValueType>
//template<typename ValueType>
__global__ void CalcuSSSP(const Vertex *csr_v,const Vertex *csr_e, ValueType *csr_w, ValueType *dis,
                          Vertex *active_vert, Vertex *active_vert_num, bool *isactive,
                          const Vertex vert_num, Vertex source,Vertex *iteration_id,int iter){

    size_t thread_id = threadIdx.x;
    size_t schedule_offset_init = blockDim.x * blockIdx.x;
    size_t vid = 0;

    while (schedule_offset_init < vert_num) {

        vid = schedule_offset_init + thread_id;

        if (vid < vert_num ) {
            dis[vid] = 99999999.0;
            isactive[vid] = false;
        }
        if (vid < 2000) {
            iteration_id[vid] = vid;
        }
        schedule_offset_init += blockDim.x * gridDim.x;
    }

    //prepare for iteration
    size_t global_id = thread_id + blockDim.x * blockIdx.x;
    if (global_id == 0) {
        *active_vert_num = 1;
        active_vert[0] = source;
        dis[source] = 0;
        g_mutex1 = 0;
        g_mutex2 = 0;
        g_mutex4 = 0;
        g_mutex5 = 0;
    }
    __threadfence();
    if (threadIdx.x == 0) {
        atomicAdd((int*) &g_mutex3, 1);
        while ((g_mutex3 == 0) || (g_mutex3 % gridDim.x) ) {}
    }
    __syncthreads();



    int l_iteration_id = 0;
    int total_avtive_num = 1;
    size_t lane_id = thread_id % THREADS_PER_WARP;
    size_t warp_id = thread_id / THREADS_PER_WARP;

    typedef cub::BlockScan<int, THREADS_PER_BLOCK> BlockScan;
    __shared__ typename BlockScan::TempStorage block_temp_storage;

    volatile __shared__ int comm[THREADS_PER_BLOCK/THREADS_PER_WARP][3];//[256/32][3]第一维是多少个warp，256/32=8，8个warp，
    volatile __shared__ ValueType commr[THREADS_PER_BLOCK/THREADS_PER_WARP];//每个warp对应一个值
    volatile __shared__ int comm2[THREADS_PER_BLOCK]; //一维数组大小256，int
    volatile __shared__ int commd2[THREADS_PER_BLOCK]; //out-degree
    volatile __shared__ ValueType commr2[THREADS_PER_BLOCK];

    //while (!(l_iteration_id > 20 && *active_verts_num < 10)) {
     while ((*active_vert_num > 0&&l_iteration_id<iter)&&(l_iteration_id <10||*active_vert_num>100) ){
        //while (l_iteration_id < 1) {
        l_iteration_id += 1;
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd((int *)&g_mutex4, 1);
            while (g_mutex4 < gridDim.x * iteration_id[l_iteration_id]) {}
        }
        __syncthreads();
        size_t schedule_offset = blockDim.x * blockIdx.x;
        size_t idx = 0;
        int row_start, row_end;
        int u, v;
        ValueType du, weight; //dis value of u

        while (schedule_offset < *active_vert_num) {
            idx = schedule_offset + thread_id;
            if (idx < *active_vert_num) {
                u = active_vert[idx];
                du = dis[u];
                row_start = csr_v[u];
                row_end = csr_v[u+1];

            } else {
                row_start = 0;
                row_end = 0;
            }

            while (__syncthreads_or((row_end-row_start)>=THREADS_PER_BLOCK)) {
                if ((row_end-row_start) >= THREADS_PER_BLOCK) {
                    comm[0][0] = thread_id; //I (thread_id) want to process the active vertex assigned to me
                }
                __syncthreads(); //all threads in one block vote to processing their own vertices

                if (comm[0][0] == thread_id) {
                    comm[0][1] = row_start; //the vertx owned by me will be processed in this <1>-while loop.
                    comm[0][2] = row_end;
                    commr[0] = du;
                    row_start = row_end;//avoid processing this vertex repeatedly in <2>&<3>-while
                }
                __syncthreads(); //all threads are ready to process the selected vertex

                size_t push_st = comm[0][1] + thread_id; //process the "push_st"-th outgoing edge at first.
                size_t push_ed = comm[0][2];

                while (__syncthreads_or(push_st<push_ed)) {
                    if (push_st < push_ed) {
                        v = csr_e[push_st]; //target vertex id
                        weight = commr[0]+csr_w[push_st];
                        ValueType old = atomicMin(dis+v, weight);
                        if(old != dis[v])
                            isactive[v] = true;

                    }
                    push_st += THREADS_PER_BLOCK;//直到u的所有外邻居被处理
                }

            }//while<0>, outdeg > 256

            ////<2> warp(32)
            while (__any_sync(FULL_MASK, (row_end-row_start)>=THREADS_PER_WARP)) {
                if ((row_end-row_start) >= THREADS_PER_WARP) {
                    comm[warp_id][0] = lane_id;
                }

                if (comm[warp_id][0] == lane_id) {
                    comm[warp_id][1] = row_start; //vertex owned by the "lane_id"-th thread in a warp is scheduled
                    comm[warp_id][2] = row_end;
                    commr[warp_id] = du;
                    row_start = row_end; //avoid processing this vertex repeatedly in <3>-while
                }
                size_t push_st = comm[warp_id][1] + lane_id; //process the "push_st"-th outgoing edge at first.
                size_t push_ed = comm[warp_id][2];

                ////<2.1>
                while (__any_sync(FULL_MASK, push_st<push_ed)) {
                    if (push_st < push_ed) {
                        v = csr_e[push_st];
                        weight = commr[warp_id] + csr_w[push_st];
                        ValueType old = atomicMin(dis+v, weight);
                        if(old != dis[v])
                            isactive[v] = true;
                    }
                    push_st += THREADS_PER_WARP; //until all outgoing edges of "u" have been processed
                }//while<2>, 处理所有outdeg > 32
            }

            //then, the out-degree of "u" is less than THREADS_PER_WARP(32)
            int thread_count = row_end - row_start;
            int deg = thread_count;
            int scatter = 0, total = 0;
            __syncthreads();
            BlockScan(block_temp_storage).ExclusiveSum(thread_count, scatter, total);
            __syncthreads();

            int progress = 0;

            ///<3>
            while (progress < total) {
                int remain = total - progress;
                while (scatter<(progress+THREADS_PER_BLOCK) && (row_start<row_end)) {
                    comm2[scatter-progress] = row_start;
                    commd2[scatter-progress] = deg;
                    commr2[scatter-progress] = du;
                    scatter++;
                    row_start++;
                }
                __syncthreads();
                int cur_batch_count = min(remain, (int)THREADS_PER_BLOCK);
                if (thread_id < cur_batch_count) {
                    v = csr_e[comm2[thread_id]];
                    weight = commr2[thread_id]+csr_w[comm2[thread_id]];
                    ValueType old = atomicMin(dis+v, weight);
                    if(old != dis[v])
                        isactive[v] = true;
                }
                __syncthreads();
                progress += THREADS_PER_BLOCK;
            }
            schedule_offset += blockDim.x * gridDim.x;
        }
        __syncthreads();
 __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd((int *)&g_mutex5, 1);
            while (g_mutex5 < gridDim.x * iteration_id[l_iteration_id]) {}
        }
        __syncthreads();
        *active_vert_num = 0;

        __threadfence();
        if (threadIdx.x == 0) {
            atomicAdd((int *) &g_mutex1, 1);
            while (g_mutex1 < gridDim.x * iteration_id[l_iteration_id]) {}
        }
        __syncthreads();

        volatile __shared__ size_t output_cta_offset;

        size_t thread_idx = threadIdx.x;
        size_t schedule_offset_barrir = blockDim.x * blockIdx.x;
        size_t vid = 0;
        while (__syncthreads_or(schedule_offset_barrir < vert_num)) {
            vid = schedule_offset_barrir + thread_idx;
            int thread_cnt = 0;
            if (vid < vert_num) {
                if (isactive[vid]) {
                    isactive[vid] = false;
                    thread_cnt = 1;
                }
            }
            int scatter = 0, total = 0;
            __syncthreads();
            BlockScan(block_temp_storage).ExclusiveSum(thread_cnt, scatter, total);
            __syncthreads();
            if (thread_id == 0) {
                output_cta_offset = atomicAdd(active_vert_num, total);
            }
            __syncthreads();
            if (thread_cnt > 0) {
                active_vert[output_cta_offset+scatter] = vid;
            }

            schedule_offset_barrir += blockDim.x * gridDim.x;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd((int *)&g_mutex2, 1);
            while (g_mutex2 < gridDim.x * iteration_id[l_iteration_id]) {}

        }
        total_avtive_num = *active_vert_num;
        __syncthreads();
    }

}

