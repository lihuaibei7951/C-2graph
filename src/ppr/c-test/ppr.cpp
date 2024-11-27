#include <iostream>
#include <queue>
#include <algorithm>
#include <time.h>
#include "ppr.h"
#include <unistd.h>
#include <sys/time.h>
#include <vector>

using namespace std;

#include "cuda_runtime.h"
#include "device_launch_parameters.h" //making threadIdx/blockIdx/blockDim/gridDim visible



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
//需要把相关信息导入进来
void initQueue(BufferQueue *buffer, int MaxSize) {

	buffer->length = 0;
	buffer->flag = 1;
	Data *data;
	data = new Data;
	ValueType *reserve, *residue;
	int *global_ft, *flag, *global_ft_cnt;
	cudaMallocHost(&global_ft_cnt, sizeof(int));
	cudaMallocHost(&reserve, sizeof(ValueType)*vert_num_new);
	cudaMallocHost(&residue, sizeof(ValueType)*vert_num_new);
	cudaMallocHost(&global_ft, sizeof(int)*vert_num_new);
	cudaMallocHost(&flag, sizeof(int));

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
		int *global_ft, *flag, *global_ft_cnt;
		cudaMallocHost(&global_ft_cnt, sizeof(int));
		cudaMallocHost(&reserve, sizeof(ValueType)*vert_num_new);
		cudaMallocHost(&residue, sizeof(ValueType)*vert_num_new);
		cudaMallocHost(&global_ft, sizeof(int)*vert_num_new);
		cudaMallocHost(&flag, sizeof(int));

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
void forward_push(vector<vector<int>> &adj,vector<ValueType> &degree, ValueType* reserve, ValueType* residue, int *global_ft_1, int global_ft1_cnt) {

	//clock_t start = clock();
	int* global_ft_2 = new int[vert_num_new];
	int global_ft2_cnt = 0;

	ValueType* messages = new ValueType[vert_num_new];
	bool* isactive = new bool[vert_num_new]();
	//if (cnt<10)cout <<  cnt++<<":	"<<"global_ft_cnt :  " << global_ft1_cnt << endl;



     int global_ft1_cnt_new=0;
	int iteration_id = 1;
	cnt++;
	while(global_ft1_cnt>0) {
//&&iteration_id<50
	//cout << "global_ft1_cnt :  " << global_ft1_cnt << endl;
		int u = 0, v = 0;
		ValueType ru = 0, deg = 0, msg = 0;
		
		for(int i = 0; i < global_ft1_cnt+global_ft1_cnt_new; i++){
			u = global_ft_1[i];
			//cout << u <<  "\t";
			ru = residue[u];
			deg = adj[u].size();
			msg = ((1 - alpha)*ru)/degree[u];
			for(int j = 0; j < deg; j++) {
				v = adj[u][j];
				messages[v] += msg;
				if(!isactive[v]) {
					global_ft_2[global_ft2_cnt++] = v;
					isactive[v] = true;
				}
			}
			residue[u] = 0;
		}

		global_ft1_cnt = 0;

		global_ft1_cnt_new=0;
		for(int i = 0; i < global_ft2_cnt; i++) {
			v = global_ft_2[i];
			residue[v] += messages[v];
			messages[v] = 0;
			isactive[v] = false;

			if (degree[v] == 0.8) {
				global_ft_1[global_ft1_cnt_new++] = v;
			}else if((residue[v]/degree[v]) >= rmax) {
				reserve[v] += (residue[v] * alpha);
				global_ft_1[global_ft1_cnt++] = v;
			}
		}

		global_ft2_cnt = 0;
		iteration_id++;
		//gettimeofday(&t2, NULL);

		//timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
		//cout <<  cnt++ << endl;
	}
	//if(global_ft1_cnt!=0) cout << global_ft1_cnt << endl;

	//cout <<  cnt++ << endl;

	delete[] global_ft_2;
	global_ft_2 = NULL;

	delete[] messages;
	messages = NULL;

	delete[] isactive;
	isactive = NULL;
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
				forward_push(graph.adj, graph.degree,Pdata->rear->reserve, Pdata->rear->residue, 
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
