#include <iostream>
#include <queue>
#include <algorithm>
#include "Graph.h"
//#include "CilkUtil.h"
#include "SSSP.h"
#include <sys/time.h>
#include <unistd.h>
#include <set>
#include "cuda_runtime.h"
#include "device_launch_parameters.h" //making threadIdx/blockIdx/blockDim/gridDim visible


Graph graph;
int cnt=0;
const ValueType alpha = 0.2f;

//insert Queuenode
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
    if(buffer->front == NULL){
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
    ValueType *distance;
    int *global_ft, *flag, *global_ft_cnt;
    cudaMallocHost(&global_ft_cnt, sizeof(int));
    cudaMallocHost(&distance, sizeof(ValueType)*graph.vert_num);
    cudaMallocHost(&global_ft, sizeof(int)*graph.vert_num);
    cudaMallocHost(&flag, sizeof(int));

    data->global_ft_cnt = global_ft_cnt;
    *(flag) = 1;
    data->distance = distance;
    data->global_ft = global_ft;
    data->flag = flag;
    data->next = NULL;
    buffer->front = data;
    buffer->rear = buffer->front;
    Data *ptr = buffer->front;
    for (int i = 0; i < MaxSize; i++) {
        Data *data;
        data = new Data;
        ValueType *distance;
        int *global_ft, *flag, *global_ft_cnt;
        cudaMallocHost(&global_ft_cnt, sizeof(int));
        cudaMallocHost(&distance, sizeof(ValueType)*graph.vert_num);
        cudaMallocHost(&global_ft, sizeof(int)*graph.vert_num);
        cudaMallocHost(&flag, sizeof(int));

        *flag = 1;
        data->global_ft_cnt = global_ft_cnt;
        data->distance = distance;
        data->global_ft = global_ft;
        data->flag = flag;
        data->next = NULL;

        ptr->next = data;
        ptr = ptr->next;
    }
    ptr->next = buffer->front;
}


//
void Bellman_ford(ValueType *distance, int *global_ft, int global_ft_cnt) {

    vector<Vertex> active_vert(global_ft, global_ft + global_ft_cnt);
    vector<Vertex> active_vertx;
    vector<bool> flag(graph.vert_num,false);
    int iter=0;
    while (!active_vert.empty()) {
    	cout<<iter++<<"----------"<<active_vert.size()<<endl;
        Vertex v = 0;
        // 顶点遍历
        for (const auto& u : active_vert) {
            if (graph.csr_v[u + 1] - graph.csr_v[u] <= 0) continue;
            Vertex start = graph.csr_v[u];
            Vertex end = graph.csr_v[u + 1];
            for (Vertex j = start; j < end; j++) {
                v = graph.csr_e[j];
                if(!flag[v]&&distance[v]>distance[u]+graph.csr_w[j]){
                    active_vertx.emplace_back(v); // 无序插入
                    flag[v]=true;
                }
                distance[v]= min(distance[v],distance[u]+graph.csr_w[j]);
            }
        }
        for(auto i:active_vertx){
        	flag[i]=false;
        }
        // 边界检测
        active_vert.resize(0);
        active_vert.swap(active_vertx);

    }
  /*  for(int i=0;i<10;i++){
        cout<<i<<" "<<distance[i]<<endl;
    }*/
}


void *SSSP_CPU(void *data) {


    BufferQueue *Pdata = (BufferQueue*)data;
    cout << "子线程已经启动" << endl;

    int cnt = 1;
    double Time = 0.0;
    double Time2 = 0;
    double Time3 = 0;

    struct timeval t1,t2;
    struct timeval t1cpu, t2cpu;
    double timeuse;
    gettimeofday(&t1, NULL);

    //double Time = 0;

    while(1) {
        //cout<<"循环执行一次"<<endl;
        if (Pdata->front == Pdata->rear) {
            if (Pdata->flag == -1) break;
            else continue;
        } else {
            if (*(Pdata->rear->flag) == -1) {
                //cout << "source = " << Pdata->rear->source << " result -----" << endl;
                gettimeofday(&t1cpu, NULL);
                Bellman_ford(Pdata->rear->distance, Pdata->rear->global_ft, *(Pdata->rear->global_ft_cnt));
                gettimeofday(&t2cpu, NULL);
                //display(Pdata->rear->distance);
                outQueue(Pdata);
                //Time += (t2cpu.tv_sec - t1cpu.tv_sec) + (double)(t2cpu.tv_usec - t1cpu.tv_usec)/1000000.0;
                Time += (t2cpu.tv_sec - t1cpu.tv_sec) + (double)(t2cpu.tv_usec - t1cpu.tv_usec)/1000000.0;
            } else {
                continue;
            }
        }
    }

    cout << "CPU real run time :\t" << Time << endl;
    gettimeofday(&t2, NULL);
    timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    cout << "timeval runtime: " << timeuse << " seconds" << endl;
    pthread_exit(NULL);
}

