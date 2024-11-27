#include <iostream>
#include <queue>
#include <algorithm>
#include <time.h>
#include "Graph.h"
#include "PPR.h"

#include <unistd.h>
#include <sys/time.h>
#include <vector>
#include <unordered_set>

using namespace std;

#include "/usr/local/cuda-11.7/include/cuda_runtime.h"
#include "/usr/local/cuda-11.7/include/device_launch_parameters.h" //making threadIdx/blockIdx/blockDim/gridDim visible


Graph graph;
const ValueType alpha = 0.2f;
int cnt=0;
//insert queuenode
void insertQueue(BufferQueue *buffer, Data *node){
    if(buffer->front == NULL){
        buffer->front = node;
    } else {
        buffer->rear->next = node;
    }
    buffer->rear = node;
}

//del queuenode
int outQueue(BufferQueue *buffer){
    Data *temp;
    if(buffer->rear == buffer->front){
        printf("队空\n");
        return 0;
    } else {
        *(buffer->rear->flag) = 1;
        buffer->rear = buffer->rear->next;
        buffer->length--;
    }
    return 1;
}

void initQueue(BufferQueue *buffer, int MaxSize) {

    buffer->length = 0;
    buffer->flag = 1;
    Data *data;
    data = new Data;
    ValueType *reserve, *residue;
    Vertex *global_ft, *flag, *global_ft_cnt;
    cudaMallocHost(&global_ft_cnt, sizeof(Vertex));
    cudaMallocHost(&reserve, sizeof(ValueType)*graph.vert_num);
    cudaMallocHost(&residue, sizeof(ValueType)*graph.vert_num);
    cudaMallocHost(&global_ft, sizeof(Vertex)*graph.vert_num);
    cudaMallocHost(&flag, sizeof(Vertex));

    data->global_ft_cnt = global_ft_cnt;
    *(flag) = 1;
    data->reserve = reserve;
    data->residue = residue;
    data->global_ft = global_ft;
    data->flag = flag;
    data->next = NULL;
    buffer->front = data;
    buffer->rear = buffer->front;
    Data *ptr = buffer->front;
    for (int i = 0; i < MaxSize; i++) {
        Data *data;
        data = new Data;
        ValueType *reserve, *residue;
        Vertex *global_ft, *flag, *global_ft_cnt;
        cudaMallocHost(&global_ft_cnt, sizeof(Vertex));
        cudaMallocHost(&reserve, sizeof(ValueType)*graph.vert_num);
        cudaMallocHost(&residue, sizeof(ValueType)*graph.vert_num);
        cudaMallocHost(&global_ft, sizeof(Vertex)*graph.vert_num);
        cudaMallocHost(&flag, sizeof(Vertex));

        *flag = 1;
        data->global_ft_cnt = global_ft_cnt;
        data->reserve = reserve;
        data->residue = residue;
        data->global_ft = global_ft;
        data->flag = flag;
        data->next = NULL;

        ptr->next = data;
        ptr = ptr->next;
    }
    ptr->next = buffer->front;
}

//print information
void display(ValueType *reserve) {
    for (int i = 0; i < 20; i++) {
        cout << i << ", " << reserve[i] << endl;
    }
    cout << "当前结果打印完毕" << endl;
}

inline ValueType AtomicAddMessage(int u, ValueType msg, ValueType *messages) {
    if (sizeof(ValueType) == 4) {
        volatile ValueType old_val, new_val;
        do {
            old_val = messages[u];
            new_val = old_val + msg;
        } while (!__sync_bool_compare_and_swap(
                (int*)(messages+u), *((int*)&old_val), *((int*)&new_val)));
        return old_val;
    } else {
        std::cout << "CAS bad length" << std::endl;
        exit(-1);
    }
}

//forward_push  chuanxing
void forward_pushold(ValueType* reserve, ValueType* residue, Vertex* global_ft_1, Vertex global_ft1_cnt) {
    //clock_t start = clock();

    ValueType* messages = new ValueType[graph.vert_num];
    Vertex iteration_id = 1;

    // 使用 unordered_set 替代 set
    std::unordered_set<Vertex> active_vert(global_ft_1, global_ft_1 + global_ft1_cnt);
    std::unordered_set<Vertex> active_vertx;
    Vertex iter = 0;

    while (!active_vert.empty() && iter < 100) {
    //	cout<<cnt<<"----------"<<active_vert.size()<<endl;
        Vertex v = 0;
        ValueType ru = 0, msg = 0, i = 0;

        // 顶点遍历
        for (const auto& u : active_vert) {
            ru = residue[u];
            residue[u] = 0;
            if (graph.csr_v[u + 1] - graph.csr_v[u] <= 0) continue;

            msg = (1 - alpha) * ru;
            Vertex start = graph.csr_v[u];
            Vertex end = graph.csr_v[u + 1];

            for (Vertex j = start; j < end; j++) {
                v = graph.csr_e[j];
                messages[v] += msg*graph.csr_w[j];
                active_vertx.insert(v); // 无序插入
            }
        }

        // 边界检测
        active_vert.clear();
        for (const auto& v : active_vertx) {
            residue[v] += messages[v];
            messages[v] = 0;

            if (residue[v] >(graph.csr_v[v + 1] - graph.csr_v[v]) *graph.rmax) {
                reserve[v] += (residue[v] * alpha);
                active_vert.insert(v); // 无序插入
            }
        }
        // 清除内容
        active_vertx.clear();
        
        
    }
//     for (int i = 0; i <=100; i++) {
//		cout<<i<<".\tpageran\t "<<reserve[i] << "\tresidual\t" <<residue[i] <<endl;
//	}

    delete[] messages;
    messages = nullptr;
    
}

//forward_push  chuanxing
void forward_push(ValueType* reserve, ValueType* residue, Vertex* global_ft_1, Vertex global_ft1_cnt) {
    //clock_t start = clock();
if(!global_ft1_cnt) return ;
    ValueType* messages = new ValueType[graph.vert_num];
    Vertex * flag = new Vertex[graph.vert_num];
    Vertex iteration_id = 1;

    // 使用 unordered_set 替代 set
    vector<Vertex> active_vert(global_ft_1, global_ft_1 + global_ft1_cnt);
    vector<Vertex> active_vertx;
    Vertex iter = 0;

    while (!active_vert.empty() && iter < 1000) {
//    	cout<<iter<<"----------"<<active_vert.size()<<endl;
        Vertex v = 0;
        ValueType ru = 0, msg = 0, i = 0;
        // 顶点遍历
        for (const auto& u : active_vert) {
            ru = residue[u];
            residue[u] = 0;
            if (graph.csr_v[u + 1] - graph.csr_v[u] <= 0) continue;
            msg = (1 - alpha) * ru;
            Vertex start = graph.csr_v[u];
            Vertex end = graph.csr_v[u + 1];

            for (Vertex j = start; j < end; j++) {
                v = graph.csr_e[j];
                messages[v] += msg*graph.csr_w[j];
                if(!flag[v]){
                    active_vertx.emplace_back(v); // 无序插入
                    flag[v]=true;
                }
            }
        }

        // 边界检测
        active_vert.resize(0);
        for (const auto& v : active_vertx) {
            residue[v] += messages[v];
            messages[v] = 0;
            flag[v]=false;

            if (residue[v] >(graph.csr_v[v + 1] - graph.csr_v[v]) *graph.rmax) {
                reserve[v] += (residue[v] * alpha);
                active_vert.emplace_back(v); // 无序插入
            }
        }
        // 清除内容
        active_vertx.resize(0);
    }
   /*     for (int i = 0; i <=2; i++) {
		cout<<i<<".\tpageran\t "<<reserve[i] << "\tresidual\t" <<residue[i] <<endl;
	}*/

    delete[] messages;
    messages = nullptr;
    delete[] flag;
    flag = nullptr;
}

void* ppr_CPU(void* data){

    BufferQueue *Pdata = (BufferQueue*)data;
//	cout << "线程已经启动"<< endl;
    //copy(graph.degree.begin(), graph.degree.end(), degree);
    //for(int i=0;i<100;i++){
    //	cout<<graph.degree[i]<<endl;
    //}
    //int cnt = 1;
    double Time = 0;
    double Time2 = 0;
    double Time3 = 0;

    struct timeval t1, t2;
    struct timeval t1cpu, t2cpu;
    double timeuse;
    gettimeofday(&t1, NULL);

    while(1){
        if(Pdata->front == Pdata->rear){
            if(Pdata->flag == -1) break;
            else continue;
        } else {
            if (*(Pdata->rear->flag) == -1) {
                gettimeofday(&t1cpu, NULL);
                forward_push(Pdata->rear->reserve, Pdata->rear->residue,
                             Pdata->rear->global_ft, *(Pdata->rear->global_ft_cnt));
                //	cout << "线程flag:" << Pdata->flag<< endl;
                gettimeofday(&t2cpu, NULL);
                outQueue(Pdata);
                Time += (t2cpu.tv_sec - t1cpu.tv_sec) + (double)(t2cpu.tv_usec - t1cpu.tv_usec)/1000000.0;
            } else {
                continue;
            }
        }

    }
    gettimeofday(&t2, NULL);
    timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
//	cout << "线程已经结束flag:" << Pdata->flag<< endl;
    cout << "timeval runtime: " << timeuse << " seconds" << endl;
    cout << "CPU real run time : " << Time << endl;

    //cout << "CPU real run time :钱钱钱钱钱钱 "  << endl;
    pthread_exit(NULL);
}
