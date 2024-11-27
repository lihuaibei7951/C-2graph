#include <iostream>
#include <queue>
#include <algorithm>
#include <time.h>
#include "Graph.h"
#include "PPR.h"
#include <unistd.h>
#include <sys/time.h>
#include <vector>
#include <set>
using namespace std;

#include "cuda_runtime.h"
#include "device_launch_parameters.h" //making threadIdx/blockIdx/blockDim/gridDim visible


string dir = "../../dataset/cnr-2000/filter";
Graph graph(dir);
const Vertex vert_num = graph.vert_num;
const Vertex edge_num = graph.edge_num;

const vector<Vertex> csr_v = graph.csr_v;
const vector<Vertex> csr_e = graph.csr_e;

//ValueType *degree = new ValueType[graph.get_number_vertices_new()];
int cnt=0;
const ValueType alpha = 0.2f;
const ValueType rmax = 0.1f*(1.0f/edge_num);

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
	cudaMallocHost(&reserve, sizeof(ValueType)*vert_num);
	cudaMallocHost(&residue, sizeof(ValueType)*vert_num);
	cudaMallocHost(&global_ft, sizeof(Vertex)*vert_num);
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
		cudaMallocHost(&reserve, sizeof(ValueType)*vert_num);
		cudaMallocHost(&residue, sizeof(ValueType)*vert_num);
		cudaMallocHost(&global_ft, sizeof(Vertex)*vert_num);
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
void forward_push(ValueType* reserve, ValueType* residue, Vertex *global_ft_1, Vertex global_ft1_cnt) {

	//clock_t start = clock();

	ValueType* messages = new ValueType[vert_num];

	Vertex iteration_id = 1;
	set<Vertex> active_vert(global_ft_1, global_ft_1 + global_ft1_cnt);
	set<Vertex> active_vertx;	
	Vertex iter = 0;
	while(active_vert.size()>0&&iter<100) {
		
		//cout<<iter++<<" : "<<active_vert.size()<<endl;
		Vertex u = 0, v = 0;
		ValueType ru = 0, msg = 0,i =0;
		
		//顶点遍历
		
		for(auto it = active_vert.begin(); it != active_vert.end(); ++it){
			u = *it;
			ru = residue[u];
			residue[u] = 0;
			if(csr_v[u+1]-csr_v[u]<=0) continue;
			msg = ((1 - alpha)*ru)/(csr_v[u+1]-csr_v[u]);
			Vertex start = csr_v[u];
			Vertex end = csr_v[u+1];
			for(Vertex j = start; j < end; j++) {
				v = csr_e[j];
				messages[v] += msg;
				active_vertx.insert(v);
		     }
		}
		
         //边界检测
         active_vert.clear();
         for(auto it = active_vertx.begin(); it != active_vertx.end(); ++it){
         	     v = *it;
		     residue[v] += messages[v];
			messages[v] = 0;
			if(residue[v]/(csr_v[v+1]-csr_v[v]) >= rmax) {
				reserve[v] += (residue[v] * alpha);
				active_vert.insert(v);
			}
         }
         	
         // 清除内容
        active_vertx.clear();

	}
	//if(global_ft1_cnt!=0) cout << global_ft1_cnt << endl;
	//cout <<  cnt++ << endl;
   /*  for (int i = 0; i <=10; i++) {
		cout<<i<<".\tpageran\t "<<reserve[i] << "\tresidual\t" <<residue[i] <<endl;
	}*/

	

	delete[] messages;
	messages = NULL;

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
bool file_exists(std::string filename) {
    struct stat st;
    return stat(filename.c_str(), &st) == 0;
}

long file_size(std::string filename) {
    struct stat st;
    assert(stat(filename.c_str(), &st) == 0);
    return st.st_size;
}
vector<Vertex> read_binary2vector(std::string filename) {
    std::vector<Vertex> out;
    if (!file_exists(filename)) {
        fprintf(stderr, "file:%s not exist.\n", filename.c_str());
        exit(0);
    }
    long fs = file_size(filename);
    long ele_cnt = fs / sizeof(Vertex);
    out.resize(ele_cnt);
    FILE *fp;
    fp = fopen(filename.c_str(), "r");
    if (fread(&out[0], sizeof(Vertex), ele_cnt, fp) < ele_cnt) {
        fprintf(stderr, "Read failed.\n");
    }
    return out;
}