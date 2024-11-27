#include <time.h>
#include "Util.cuh"
#include <cuda_runtime.h>
#include <sys/time.h>
#include "Graph.h"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>


__device__ volatile int g_mutex1;
__device__ volatile int g_mutex2;
__device__ volatile int g_mutex3;
__device__ volatile int g_mutex4;
__device__ volatile int g_mutex5;

__global__ void calcuatePPR(const ll *csr_v, const Vertex *csr_e, const ValueType *csr_w,
                            bool *f1, Vertex *act, Vertex *act_num, ValueType *pi, ValueType *oldval, ValueType *newval,
                            int *iteration_id,int *maxiter, int *record,
                            const int vert_num,const int source,const ValueType alpha
);


int main(int argc, char **argv) {
    // Initialize graph data in host & device memory
    cudaFree(0);
    // 获取命令行参数
    std::string dir = argv[1];

    // 获取，csr_v ,csr_e ,v_r,degree,order;
    Graph graph(dir);
    std::cout << "test for study how to use cuda" << endl;
    int vert_num = graph.vert_num;
    int edge_num = graph.edge_num;
    int source = 1;

    thrust::device_vector<ll> d_csr_v(vert_num+1, 0);
    thrust::device_vector<Vertex> d_csr_e(edge_num,0);
    thrust::device_vector<ValueType> d_csr_w(edge_num,0);

    thrust::device_vector<bool> d_f1(vert_num, false);
    thrust::device_vector<Vertex> d_act(vert_num, 0);
    thrust::device_vector<Vertex> d_f(1,source);

    thrust::device_vector<ValueType> d_pi(vert_num,0.0);
   // thrust::host_vector<ValueType> h_pi(vert_num,0.0);
    thrust::device_vector<ValueType> d_oldval(vert_num,0);
    thrust::device_vector<ValueType> d_newval(vert_num,0);

    thrust::copy(graph.csr_v.begin(), graph.csr_v.end(), d_csr_v.begin());
    thrust::copy(graph.csr_e.begin(), graph.csr_e.end(), d_csr_e.begin());
    thrust::copy(graph.csr_w.begin(), graph.csr_w.end(), d_csr_w.begin());


    int *iteration_id;
    cudaMalloc(&iteration_id, sizeof(int)*500);

    int *record;
    cudaMalloc(&record, sizeof(int)*500);

    int *maxiter;
    cudaMalloc(&maxiter, sizeof(int)*1);

    struct timeval t_start, t_stop;
    double timeuse;
    gettimeofday(&t_start, NULL);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cout << "\n==================== PPR with FORWARD PUSH starts ====================" << endl;


    int cnt = 0;
    while(1){

        cnt++;
        calcuatePPR<<<MAX_BLOCKS_NUM, THREADS_PER_BLOCK>>>(
                d_csr_v.data().get(), d_csr_e.data().get(), d_csr_w.data().get(),
                d_f1.data().get(), d_act.data().get(), d_f.data().get(),d_pi.data().get(), d_oldval.data().get(), d_newval.data().get(),
                iteration_id,maxiter,record,
                vert_num, source,alpha
        );

        if(cnt==1){
            break;
        }
    }


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float runtime = 0; //milliseconds
    cudaEventElapsedTime(&runtime, start, stop);

    cout << "gpu runtime: " << runtime/1000.0 << " seconds" << endl;
    cout << "==================== PPR with FORWARD PUSH ends ====================\n" << endl;
    int *iteration_num = new int[1];
    int *iteration_act_num = new int[500];
    ValueType *h_pi = new ValueType[500];

    CUDA_ERROR(cudaMemcpy(iteration_num, maxiter, sizeof(int)*1, cudaMemcpyDeviceToHost));
    cout << "flag 已设置成 -1  终止条件以满足		iteration_num："<<iteration_num[0]<<endl;
    CUDA_ERROR(cudaMemcpy(iteration_act_num, record, sizeof(int)*500, cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(h_pi, d_pi.data().get(), sizeof(ValueType)*500, cudaMemcpyDeviceToHost));

    cout << "0	act_num：1"<<endl;
    for(int i = 1 ;i<50;i++){
        if(iteration_act_num[i]==0) break;
        cout <<i<< "	act_num："<<iteration_act_num[i]<<"---"<<h_pi[i]<<endl;

    }


    //cout << "内存开辟耗时: " << timeMalloc << endl;
    gettimeofday(&t_stop, NULL);

    timeuse = (t_stop.tv_sec - t_start.tv_sec) + (double)(t_stop.tv_usec - t_start.tv_usec)/1000000.0;
    //cout << "main total timeval runtime: " << timeuse << " seconds" << endl;
    return 0;
}



__global__ void calcuatePPR(const ll *csr_v, const Vertex *csr_e, const ValueType *csr_w,
                            bool *f1, Vertex *act, Vertex *act_num, ValueType *pi, ValueType *oldval, ValueType *newval,
                            int *iteration_id,int *maxiter, int *record,
                            const int vert_num,const int source,const ValueType alpha
){
    size_t thread_id = threadIdx.x;
    size_t schedule_offset_init = blockDim.x * blockIdx.x;
    size_t vid = 0;
    while (schedule_offset_init < vert_num) {
        vid = schedule_offset_init + thread_id;
        if(vid<vert_num){
            pi[vid]=0.0;
            newval[vid]=0.0;
            f1[vid]=false;
          //  if(vid<=source){
             //   oldval[vid]=10.0;
             //   act[vid]=vid;
           // }else{
                oldval[vid]=0.0;
           // }
        }
        if (vid < 500) {
            iteration_id[vid] = vid;
            record[vid]=0;
        }
        schedule_offset_init += blockDim.x * gridDim.x;
    }
    //prepare for the 1st iteration
    size_t global_id = thread_id + blockDim.x*blockIdx.x;
    if (global_id == 0) {
        *act_num=1;
        oldval[source]=10.0;
        act[0]=source;
        g_mutex1 = 0;
        g_mutex2 = 0;
        g_mutex4 = 0;
        g_mutex5 = 0;
        //printf("source =  初始化完成\n");
    }

    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        atomicAdd((int*) &g_mutex3, 1);
        while ((g_mutex3 == 0) || (g_mutex3 % gridDim.x) ) {}
    }
    __syncthreads();

    int l_iteration_id = 0;
    int total_active_verts_num = *act_num;
    size_t lane_id = thread_id % THREADS_PER_WARP;
    size_t warp_id = thread_id / THREADS_PER_WARP; //the i-th warp (from 0)  当前块内warp的id
    typedef cub::BlockScan<int, THREADS_PER_BLOCK> BlockScan;
    __shared__ typename BlockScan::TempStorage block_temp_storage;
    volatile __shared__ int comm[THREADS_PER_BLOCK/THREADS_PER_WARP][3];//[256/32][3]第一维是多少个warp，256/32=8，8个warp，
    volatile __shared__ ValueType commr[THREADS_PER_BLOCK/THREADS_PER_WARP];//每个warp对应一个值
    volatile __shared__ int comm2[THREADS_PER_BLOCK]; //一维数组大小256，int
    volatile __shared__ ValueType commd2[THREADS_PER_BLOCK]; //out-degree
    volatile __shared__ ValueType commr2[THREADS_PER_BLOCK];

    while (total_active_verts_num>0&&l_iteration_id<400) {
        __syncthreads();
        l_iteration_id += 1;

        __syncthreads();
        __threadfence();
        if (thread_id == 0) {
            atomicAdd((int*) &g_mutex4, 1);
            while (g_mutex4 < gridDim.x * iteration_id[l_iteration_id]) {}

        }
        __syncthreads();
        size_t schedule_offset = blockDim.x * blockIdx.x;
        size_t idx = 0;
        int u = 0,v=0,w=0;
        int row_start, row_end;
        ValueType ru, msg;
        while (schedule_offset < total_active_verts_num) {
            idx = schedule_offset + thread_id;
            if (idx < total_active_verts_num) {
                u = act[idx];
                ru=oldval[u];
                row_start = csr_v[u]; //start offset of outgoing edges of "u" in "col_ind"
                row_end = csr_v[u + 1]; //end offset of outgoing edges of "u" in "col_ind" (exclusive)
            } else {
                row_start = 0;
                row_end = 0;
            }
            //while(1)
            while (__syncthreads_or((row_end - row_start) >= THREADS_PER_BLOCK)) {
                if ((row_end - row_start) >= THREADS_PER_BLOCK) {
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
                while (__syncthreads_or(push_st < push_ed)) {
                    if (push_st < push_ed) {
                        v = csr_e[push_st]; //target vertex id
                        msg=csr_w[push_st]*commr[0];
                        atomicAdd(newval + v, msg);
                        f1[v]=true;


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
                        msg=csr_w[push_st]*commr[warp_id];
                        atomicAdd(newval + v, msg);
                        f1[v]=true;
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
                    commr2[scatter-progress] = ru;
                    scatter++;
                    row_start++;
                }
                __syncthreads();
                int cur_batch_count = min(remain, (int)THREADS_PER_BLOCK); //how many threads are required?
                if (thread_id < cur_batch_count) {
                    v = csr_e[comm2[thread_id]];
                    msg=csr_w[comm2[thread_id]]*commr2[thread_id];
                    atomicAdd(newval + v, msg);
                    f1[v]=true;
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
            atomicAdd((int*) &g_mutex5, 1);
            while (g_mutex5 < gridDim.x * iteration_id[l_iteration_id]) {}

        }
        __syncthreads();
       *act_num = 0;
        __syncthreads();
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
        ValueType delta=0.0;
        while (__syncthreads_or(schedule_offset_barrir < vert_num)) {
            vid = schedule_offset_barrir + thread_idx;
            int thread_cnt = 0;
            if (vid < vert_num) {
                if (f1[vid]) {
                    f1[vid] = false;
                    if (newval[vid]/(csr_v[vid+1]-csr_v[vid]) > 0.0001) {//执行边界检测标准，符合条件将标志位设>置为1
                        delta=alpha*newval[vid];
                        newval[vid]=0.0;
                        oldval[vid]=delta;
                        pi[vid]+=delta;
                        thread_cnt = 1;
                    }
                }
            }
            int scatter = 0, total = 0;

            __syncthreads();
            BlockScan(block_temp_storage).ExclusiveSum(thread_cnt, scatter, total);
            __syncthreads();
            if (thread_id == 0) {
                output_cta_offset = atomicAdd(act_num, total); //run per block
            }
            __syncthreads();
            if (thread_cnt > 0) {
                act[output_cta_offset+scatter] = vid;
            }
            schedule_offset_barrir += blockDim.x * gridDim.x;
        }

        __syncthreads();
        __threadfence();
        if (thread_id == 0) {
            atomicAdd((int*) &g_mutex2, 1);
            while (g_mutex2 < gridDim.x * iteration_id[l_iteration_id]) {}
            maxiter[0] = l_iteration_id;
            if(l_iteration_id<500)	record[l_iteration_id]= *act_num;

        }
        __syncthreads();
        __threadfence();
        total_active_verts_num = *act_num;
        __syncthreads();

    }



}

