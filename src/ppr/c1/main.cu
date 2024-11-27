#include "Util.cuh"
#include <time.h>
#include "deviceMemory.cuh"
#include <cuda_runtime.h>
#include <sys/time.h>

__device__ volatile int g_mutex1;
__device__ volatile int g_mutex2;
__device__ volatile int g_mutex3;
__device__ volatile int g_mutex4; // rule 遍历
__device__ volatile int g_mutex5; // rule 消息传播
__device__ volatile int g_mutex6; // rule 消息传播
__device__ volatile int g_mutex7; // rule 消息传播


template <typename ValueType>
__global__ void calcuatePPR(const int *csr_v, const int *csr_e, ValueType *pagerank,
            ValueType *residual, ValueType *messages, int *active_vert,
            int *active_verts_num, bool *isactive, int *degree, int *csr_o, int *csr_r,const int vert_num,
            const int rule_num, const ValueType alpha, const ValueType rmax,
            int source, int *iteration_id, int max_step,int *iteration_num,int *iteration_act_num,int *iter);
__global__ void printHello() {
    // 每个线程打印一次 "Hello"
    printf("Hello\n");
}

// Dump results
void DumpResults(const int verts_num, ValueType *d_pagerank,
                 ValueType *d_residual, ValueType *d_messages);

int main(int argc, char **argv) {
    // Initialize graph data in host & device memory
    cudaFree(0);
    // 获取命令行参数
    std::string dir = argv[1];

    // 获取，csr_v ,csr_e ,v_r,degree,order;
    Graph graph(dir);
    if(graph.max_step<1) return 0;

    DeviceMemory device_memory(graph.vert_num, graph.rule_num, graph.edge_num,
                               graph.max_step);
    

    device_memory.CudaMemcpyGraph(graph);
    //std::cout << "test for study how to use cuda" << endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int *iteration_id;
    cudaMalloc(&iteration_id, sizeof(int) * 1000);
    int *iter;
    cudaMalloc(&iter, sizeof(int) * (graph.max_step+2));
    int *iteration_num = new int[1];
	int *iteration_act_num = new int[1000];
	
	CUDA_ERROR(cudaMemset(device_memory.active_verts_numStream, 0, sizeof(int)));//memset(指针， 初始值，大小）初始化
    // Initialize parameters for PPR
    int source = 0; // 101569
    ValueType alpha = 0.2f;
    ValueType rmax =0.01f * (1.0f / graph.origin_edge); // 这一步条件变了,应该是原始边数目

    struct timeval t_start, t_stop;
    double timeuse;
    gettimeofday(&t_start, NULL);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    //cout << "\n==================== PPR with FORWARD PUSH starts ""====================" << endl;
    //cout<<graph.vert_num<<"  "<<graph.rule_num<<endl;
    int cnt = 0;
    
     while (1) {
          if(graph.degree[source]==0){
			++source;
			continue;
		}
		
		cnt++;
		if(cnt%10==0){
		cudaDeviceSynchronize();
		//cout << "source = " << source << "start ------------" <<cnt<< endl;

		}
         calcuatePPR<ValueType><<<MAX_BLOCKS_NUM, THREADS_PER_BLOCK>>>(
            device_memory.csr_v, device_memory.csr_e,
            device_memory.pagerankStream, device_memory.residualStream,
            device_memory.messagesStream, device_memory.active_vert,
            device_memory.active_verts_numStream,device_memory.isactive,device_memory.degree, 
            device_memory.csr_o, device_memory.csr_r,
            graph.vert_num,graph.rule_num, alpha, rmax, source++, iteration_id,
            graph.max_step,device_memory.iteration_num,device_memory.iteration_act_num,iter);
            if (cnt == 1) {
        	      cudaDeviceSynchronize();
        	      //cout << "source = " << source << "end ------------" <<cnt<< endl;
                break;
            }
    }
    
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float runtime = 0; //milliseconds
	cudaEventElapsedTime(&runtime, start, stop);
	cout << "gpu runtime: " << runtime/1000.0 << "： seconds" << endl;

	
	CUDA_ERROR(cudaMemcpy(iteration_num, device_memory.iteration_num, sizeof(int)*1, cudaMemcpyDeviceToHost));
	cout << "flag 已设置成 -1  终止条件以满足		iteration_num："<<iteration_num[0]<<endl;
	CUDA_ERROR(cudaMemcpy(iteration_act_num, device_memory.iteration_act_num, sizeof(int)*1000, cudaMemcpyDeviceToHost));
	
	cout << "0	:act_num: 1"<<endl;
	for(int i = 1 ;i<1000  ;i++){
		if(iteration_act_num[i]==0) break;
		cout <<i<< "	:act_num:	"<<iteration_act_num[i]<<endl;
		
	}
	cout << "==================== PPR with FORWARD PUSH ends ====================\n" << endl;

	//cout << "内存开辟耗时: " << timeMalloc << endl;
	gettimeofday(&t_stop, NULL);
	DumpResults(graph.vert_num, device_memory.pagerankStream, device_memory.residualStream, device_memory.messagesStream);

	timeuse = (t_stop.tv_sec - t_start.tv_sec) + (double)(t_stop.tv_usec - t_start.tv_usec)/1000000.0;
	//cout << "main total timeval runtime: " << timeuse << " seconds" << endl;
	return 0;

}

template <typename ValueType>
__global__ void calcuatePPR(const int *csr_v, const int *csr_e, ValueType *pagerank,
            ValueType *residual, ValueType *messages, int *active_vert,
            int *active_verts_num, bool *isactive, int *degree, int *csr_o, int *csr_r,const int vert_num,
            const int rule_num, const ValueType alpha, const ValueType rmax,
            int source, int *iteration_id, int max_step,int *iteration_num,int *iteration_act_num,int *iter) {

    size_t thread_id = threadIdx.x;
    size_t schedule_offset_init = blockDim.x * blockIdx.x;
    size_t vid = 0;
    while (schedule_offset_init < vert_num + rule_num) {
        vid = schedule_offset_init + thread_id;
        // in the last batch, some threads are idle
        if (vid < vert_num + rule_num) {
            pagerank[vid] = 0;
            residual[vid] = 0;
            messages[vid] = 0;
            isactive[vid] = false;
        }

        if (vid < 1000) {
            iteration_id[vid] = vid;
            
        }
        if(vid<max_step + 1){
        		iter[vid] = 0;
        }
        schedule_offset_init +=blockDim.x * gridDim.x; // 块线程数目*网格块数=网格线程数目
    }
    __syncthreads(); // 确保第一个初始化过程内的线程块同步


    // prepare for the 1st iteration
    size_t global_id = thread_id + blockDim.x * blockIdx.x;
    if (global_id == 0) { 
    // 每一个块中线程为0的id，source=1,第一个顶点的residual值初始化为1，
        residual[source] = 1;
        *active_verts_num = 1;
        active_vert[0] = source; // 当前迭代中的活跃顶点
        pagerank[source] += alpha * residual[source];
        g_mutex1 = 0;
        g_mutex2 = 0;
        g_mutex4 = 0;
        g_mutex5 = 0;
        g_mutex6 = 0;
        g_mutex7 = 0;
        //g_mutex3 = 0;
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
    int vaddr = vert_num + rule_num;

    size_t lane_id = thread_id % THREADS_PER_WARP; // warp内线程的id
    size_t warp_id =thread_id / THREADS_PER_WARP; // the i-th warp (from 0) 当前块内warp的id

    typedef cub::BlockScan<int, THREADS_PER_BLOCK> BlockScan;

    __shared__ typename BlockScan::TempStorage block_temp_storage;

    volatile __shared__ int comm[THREADS_PER_BLOCK / THREADS_PER_WARP][3];
    //[256/32][3]第一维是多少个warp，256/32=8，8个warp，

    volatile __shared__ ValueType commr[THREADS_PER_BLOCK / THREADS_PER_WARP]; // 每个warp对应一个值

    volatile __shared__ ValueType commd[THREADS_PER_BLOCK / THREADS_PER_WARP]; // 每个warp对应一个值

    volatile __shared__ int comm2[THREADS_PER_BLOCK]; // 一维数组大小256，int

    volatile __shared__ ValueType commd2[THREADS_PER_BLOCK]; // out-degree

    volatile __shared__ ValueType commr2[THREADS_PER_BLOCK];
    while (*active_verts_num > 0&&l_iteration_id<200) { //*active_verts_num > 0       l_iteration_id<1

        l_iteration_id += 1;
        // pushmessages 当前活跃顶点发消息

        __syncthreads();
        int total_active_verts_num = *active_verts_num;
        
        __syncthreads();
	        __threadfence();
        if (thread_id == 0) {
            atomicAdd((int *)&g_mutex1, 1);
            while (g_mutex1 < gridDim.x * iteration_id[l_iteration_id]) {}
            
        }
        	__threadfence();
        
        size_t idx = 0;
        int u = 0;
        int e_start, e_end;
        int v, du;
        ValueType ru, msg;
        
            // 顶点遍历
            size_t schedule_1 = blockDim.x * blockIdx.x; // 块索引
            while (schedule_1 < total_active_verts_num) {
                idx = schedule_1 + thread_id;
                if (idx < total_active_verts_num) {
                    u = active_vert[idx];
                    ru = residual[u];
                    du = degree[u];
                    residual[u] = 0;
                    e_start = csr_v[u]; // start offset of outgoing edges of "u"
                                        // in "col_ind"
                    e_end = csr_v[u + 1]; // end offset of outgoing edges of "u"
                                          // in "col_ind" (exclusive)
                    //补全0出度点的发送方向为源顶点
                    if(csr_v[u]==csr_v[u + 1]) {
					e_start = csr_v[source];
					e_end = csr_v[source + 1];
					du = degree[source];
					
                    }
                } else {
                    e_start = 0;
                    e_end = 0;
                }
                // while(1)
                while ( __syncthreads_or((e_end - e_start) >= THREADS_PER_BLOCK)) {
                    if ((e_end - e_start) >= THREADS_PER_BLOCK) {
                        comm[0][0] =
                            thread_id; // I (thread_id) want to process the
                                       // active vertex assigned to me.
                    }
                    __syncthreads(); // all threads in one block vote to
                                     // processing their own vertices

                    if (comm[0][0] == thread_id) {
                        comm[0][1] = e_start; // the vertx owned by me will be
                                     // processed in this <1>-while loop.
                        comm[0][2] = e_end;
                        commr[0] = ru; // ru是u的残差
                        commd[0] = du; // du是u的真实度

                        e_start = e_end; // avoid processing this vertex
                                         // repeatedly in <2>&<3>-while
                    }
                    __syncthreads(); // all threads are ready to process the
                                     // selected vertex

                    size_t push_st = comm[0][1] + thread_id; // process the "push_st"-th
                                                // outgoing edge at first.
                    size_t push_ed = comm[0][2];

                    // <1.1>-while: block-granularity-outgoing edges
                    while (__syncthreads_or(push_st < push_ed)) {
                        if (push_st < push_ed) {
                            v = csr_e[push_st]; // target vertex id
                            msg = ((1 - alpha) * commr[0]) / commd[0]; // outdeg of the selected s, not "u"
                            atomicAdd(messages + v, msg);
                            isactive[v] =true;
                            
                            
                        }
                        push_st +=THREADS_PER_BLOCK; // until all outgoing edges of
                                               // "u" have been processed
                    }
                } // until all source vertices with
                  // "todo_edges_num>=THREADS_PER_BLOCK" have been processed

                // while(2)
                while (__any_sync(FULL_MASK,(e_end - e_start) >= THREADS_PER_WARP)) {
                    if ((e_end - e_start) >= THREADS_PER_WARP) {
                        comm[warp_id][0] = lane_id; // threads in the "warp_id"-th warp try to
                                     // vote
                    }
                    if (comm[warp_id][0] == lane_id) {
                        comm[warp_id][1] = e_start; // vertex owned by the "lane_id"-th thread
                                     // in a warp is scheduled
                        comm[warp_id][2] = e_end;
                        commr[warp_id] = ru;
                        commd[warp_id] = du;
                        e_start = e_end; // avoid processing this vertex
                                         // repeatedly in <3>-while
                    }
                    size_t push_st = comm[warp_id][1] + lane_id; // process the "push_st"-th
                                                    // outgoing edge at first.
                    size_t push_ed = comm[warp_id][2];

                    // <2.1>-while: warp-granularity-outgoing edges
                    while (__any_sync(FULL_MASK, push_st < push_ed)) {
                        if (push_st < push_ed) {
                            v = csr_e[push_st];
                            msg = ((1 - alpha) * commr[warp_id]) /(commd[warp_id]);
                            atomicAdd(messages + v, msg);
                            isactive[v] =true;
                        }
                        push_st +=THREADS_PER_WARP; // until all outgoing edges of
                                              // "u" have been processed
                    }
                } // until all source vertices with
                  // "todo_edges_num>=THREADS_PER_WARP" have been processed

                // while(3) then, the out-degree of "u" is less than
                // THREADS_PER_WARP(32)
                int thread_count = e_end - e_start;
                int scatter = 0, total = 0;

                __syncthreads();
                BlockScan(block_temp_storage).ExclusiveSum(thread_count, scatter, total); //
                __syncthreads(); // there are "total" edges left in every block
                int progress = 0;

                while (progress < total) {
                    int remain = total - progress;
                    while (scatter < (progress + THREADS_PER_BLOCK) &&
                           (e_start < e_end)) {
                        comm2[scatter - progress] = e_start; // 存U有的外邻居
                        commd2[scatter - progress] = du;     //
                        commr2[scatter - progress] = ru;
                        scatter++;
                        e_start++;
                    }
                    __syncthreads();
                    int cur_batch_count = min(remain, (int)THREADS_PER_BLOCK);
                    // how many threads are required?
                    if (thread_id < cur_batch_count) {
                        v = csr_e[comm2[thread_id]]; //!!!
                        msg = ((1 - alpha) * commr2[thread_id]) / commd2[thread_id];
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
///        

        // 规则遍历
        int local_iter = 0,total_rule =0,strat_r =0;
        while (local_iter < max_step ) {
        
            local_iter++;
            __syncthreads();
            
            // 同步
             strat_r = csr_o[local_iter-1];
            total_rule = csr_o[local_iter]-csr_o[local_iter-1];
            __syncthreads();
            __threadfence();
           if (thread_id == 0) {
            
                atomicAdd((int *)&g_mutex5, 1);
                while (g_mutex5 <gridDim.x *(iteration_id[l_iteration_id - 1] * max_step +local_iter)) {}
            }
            
			
            __syncthreads();
            

            
                // barrier 将新的活跃顶点加入边界
                volatile __shared__ size_t output_cta_offset;
                thread_id = threadIdx.x;
                size_t schedule_2 = blockDim.x * blockIdx.x;
                size_t vid = 0;

                while (__syncthreads_or(schedule_2 < total_rule)) {
                    // 这个函数返回的是一个布尔值，表示所有线程块中至少有一个线程满足条件。
                    vid = schedule_2 + thread_id;

                    int thread_cnt = 0;
                    if (vid < total_rule) {
                    
                        if (isactive[csr_r[vid + strat_r]]) {

                            //residual[csr_r[vid + strat_r]] =messages[csr_r[vid + strat_r]];
                            // 用残差值来传递消息
                            //messages[csr_r[vid + strat_r]] =0;
                            // 消息值置0，保证下一次迭代没问题
                            isactive[csr_r[vid + strat_r]] = false;
                            thread_cnt = 1;
                        }
                    }
                    int scatter = 0, total = 0;

                    __syncthreads();
                    BlockScan(block_temp_storage).ExclusiveSum(thread_cnt, scatter, total);
                    __syncthreads();
                    if (thread_id == 0) {
                        output_cta_offset = atomicAdd(&iter[local_iter-1], total);
                        // run per block
                    }
                    __syncthreads();
                    if (thread_cnt > 0) {
                        active_vert[output_cta_offset + scatter] = csr_r[vid + csr_o[local_iter-1]];
                        
                    }
                    schedule_2 += blockDim.x * gridDim.x;
                    // 用于更新 schedule_offset_barrir
                    // 变量的值，并且在每个线程块内都会执行。
                } // 更新规则结束

                __syncthreads();
                total_active_verts_num = iter[local_iter-1];
                __threadfence();

                if (thread_id == 0) {
                    atomicAdd((int *)&g_mutex4, 1);
                    while (g_mutex4 <gridDim.x * (iteration_id[l_iteration_id - 1] * max_step  + local_iter)) {}
                    
                    
                }
                
                __syncthreads();



            //*active_verts_num 已经更新完毕，
            // 下一步是消息传递，不需要边界检测

            	 thread_id = threadIdx.x;
                size_t schedule_3 = blockDim.x * blockIdx.x; // 块索引
				
                   size_t idxx = 0;
                    
        			int u = 0;
        			int e_start, e_end;
        			int v, du;
        			ValueType ru, msg;

                while (schedule_3 < iter[local_iter-1]) {
                    
                    idxx = schedule_3 + thread_id;
                    
                    // printf("uuuu%d\n",thread_id);
                    if (idxx < iter[local_iter-1]) {
                        u = active_vert[idxx];
                        ru = messages[u];
                        messages[u] = 0;
                        e_start = csr_v[u]; // start offset of outgoing edges of
                                            // "u" in "col_ind"
                        e_end = csr_v[u + 1]; // end offset of outgoing edges of
                                              // "u" in "col_ind" (exclusive)
                    } else {
                        e_start = 0;
                        e_end = 0;
                    }
                    // while(1)
                    while (__syncthreads_or((e_end - e_start) >=THREADS_PER_BLOCK)) {
                        if ((e_end - e_start) >= THREADS_PER_BLOCK) {
                            comm[0][0] =thread_id; // I (thread_id) want to process the
                                           // active vertex assigned to me.
                        }
                        __syncthreads(); // all threads in one block vote to
                                         // processing their own vertices

                        if (comm[0][0] == thread_id) {
                            comm[0][1] = e_start; // the vertx owned by me will be
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
                                //printf("zzzzzz%d \n", v);
                                msg = commr[0]; // outdeg of the selected s, not
                                                // "u"
                                atomicAdd(messages + v, msg);
                                isactive[v] =true;
                            }
                            push_st +=THREADS_PER_BLOCK; // until all outgoing edges
                                                   // of "u" have been processed
                        }
                    } // until all source vertices with
                      // "todo_edges_num>=THREADS_PER_BLOCK" have been processed

                    // while(2)
                    while (__any_sync(FULL_MASK, (e_end - e_start) >= THREADS_PER_WARP)) {
                        if ((e_end - e_start) >= THREADS_PER_WARP) {
                            comm[warp_id][0] =lane_id; // threads in the "warp_id"-th warp try
                                         // to vote
                        }
                        if (comm[warp_id][0] == lane_id) {
                            comm[warp_id][1] =  e_start; // vertex owned by the "lane_id"-th
                                         // thread in a warp is scheduled
                            comm[warp_id][2] = e_end;
                            commr[warp_id] = ru;
                            e_start = e_end; // avoid processing this vertex
                                             // repeatedly in <3>-while
                        }
                        size_t push_st = comm[warp_id][1] + lane_id; // process the "push_st"-th
                                                  // outgoing edge at first.
                        size_t push_ed = comm[warp_id][2];

                        // <2.1>-while: warp-granularity-outgoing edges
                        while (__any_sync(FULL_MASK, push_st < push_ed)) {
                            if (push_st < push_ed) {

                                v = csr_e[push_st];
                                // printf("xxxxxx%d \n",v);
                                msg = commr[warp_id];
                                atomicAdd(messages + v, msg);
                                isactive[v] =true;
                            }
                            push_st += THREADS_PER_WARP; // until all outgoing edges of
                                                  // "u" have been processed
                        }
                    } // until all source vertices with
                      // "todo_edges_num>=THREADS_PER_WARP" have been processed

                    // while(3) then, the out-degree of "u" is less than
                    // THREADS_PER_WARP(32)
                    int thread_count = e_end - e_start;
                    int scatter = 0, total = 0;

                    __syncthreads();
                    BlockScan(block_temp_storage)
                        .ExclusiveSum(thread_count, scatter, total); //
                    __syncthreads(); // there are "total" edges left in every
                                     // block
                    int progress = 0;

                    while (progress < total) {
                        int remain = total - progress;
                        while (scatter < (progress + THREADS_PER_BLOCK) &&(e_start < e_end)) {
                            comm2[scatter - progress] =
                                e_start; // 存U有的外邻居
                            commr2[scatter - progress] = ru;
                            scatter++;
                            e_start++;
                        }
                        __syncthreads();
                        int cur_batch_count =
                            min(remain, (int)THREADS_PER_BLOCK);
                        // how many threads are required?
                        if (thread_id < cur_batch_count) {
                            v = csr_e[comm2[thread_id]]; //!!!

                            msg = commr2[thread_id];
                            atomicAdd(messages + v, msg);
                            isactive[v] =true;
                        }
                        __syncthreads();
                        progress += THREADS_PER_BLOCK;
                    }
                    // schedule (blockDim.x * gridDim.x) active vertices per
                    // <0>-while loop
                    schedule_3 += blockDim.x * gridDim.x;
                } // 规则消息传播结束
            


        } // 规则遍历结束
///
__threadfence();
       if (thread_id == 0) {
            atomicAdd((int *)&g_mutex6, 1);
            while (g_mutex6 < gridDim.x * iteration_id[l_iteration_id]) {
            }
        }
        __syncthreads();

        // 边界检测
        __syncthreads();
        *active_verts_num = 0;
        size_t schedule_offset_rinit = blockDim.x * blockIdx.x;
     size_t rvid = 0;
    while (schedule_offset_rinit < max_step + 1) {
        rvid = schedule_offset_rinit + thread_id;
        // in the last batch, some threads are idle
        if(rvid<max_step + 1 ){
        iter[rvid] = 0;
        }

        schedule_offset_rinit +=blockDim.x * gridDim.x; 
        // 块线程数目 * 网格块数 = 网格线程数目
    }
    
        __threadfence();
        if (thread_id == 0) {
            atomicAdd((int *)&g_mutex7, 1);
            while (g_mutex7 < gridDim.x * iteration_id[l_iteration_id]) {
            }
        }
        __syncthreads();

        // barrier 将新的活跃顶点加入边界
        volatile __shared__ size_t output_cta_offset;
        size_t thread_id = threadIdx.x;
        schedule_1 = blockDim.x * blockIdx.x;
        size_t vid = 0;

        while (__syncthreads_or(schedule_1 < vert_num)) {
            // 这个函数返回的是一个布尔值，表示所有线程块中至少有一个线程满足条件。
            vid = schedule_1 + thread_id;
            int thread_cnt = 0;
            if (vid < vert_num) {
                if (isactive[vid]) {
                    residual[vid] += messages[vid];
                    messages[vid] = 0;
                    isactive[vid] = false;
                    if (residual[vid] / (degree[vid]) >= rmax) {
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
                output_cta_offset =atomicAdd(active_verts_num, total); // run per block
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
            while (g_mutex2 < gridDim.x * iteration_id[l_iteration_id]) {}
            
        }
        	__threadfence();
        if (thread_id == 0) {
               iteration_num[0] = l_iteration_id;
			if(l_iteration_id<1000)	iteration_act_num[l_iteration_id]= *active_verts_num;
        }
        __syncthreads();
        //l_active_verts_num = *active_verts_num;
        __syncthreads();

    } // while (*active_verts_num != 0);
}



// Dump result
void DumpResults(const int verts_num, ValueType *d_pagerank,
                 ValueType *d_residual, ValueType *d_messages) {
    ValueType *h_pagerank = new ValueType[verts_num];
    ValueType *h_residual = new ValueType[verts_num];
    ValueType *h_messages = new ValueType[verts_num];

    CUDA_ERROR(cudaMemcpy(h_pagerank, d_pagerank, verts_num * sizeof(ValueType),
                          cudaMemcpyDeviceToHost));

    CUDA_ERROR(cudaMemcpy(h_residual, d_residual, verts_num * sizeof(ValueType),
                          cudaMemcpyDeviceToHost));

    CUDA_ERROR(cudaMemcpy(h_messages, d_messages, verts_num * sizeof(ValueType),
                          cudaMemcpyDeviceToHost));

    ofstream outfile("/home/lhb/cucode/PPR/Judge/iter_2.txt");

    for (int i = 0; i <= 10; i++) {
        cout << i << ".\tpagerank\t " << h_pagerank[i] << "\tresidual\t"
             << h_residual[i] << endl;
    }

    for (int i = 0; i < verts_num; i++) {
        outfile << i;
        outfile << " ";
        outfile << h_pagerank[i];
        outfile << " ";
        outfile << h_residual[i];
        outfile << "\n";
    }

    delete[] h_residual;
    delete[] h_pagerank;
    delete[] h_messages;
    h_residual = NULL;
    h_pagerank = NULL;
    h_messages = NULL;
}
