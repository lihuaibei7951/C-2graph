#include <iostream>
#include <queue>
#include <algorithm>
#include "Graph.h"
#include "CilkUtil.h"
#include "SSSP.h"
#include <sys/time.h>
#include <unistd.h>
#include <set>
#include "cuda_runtime.h"
#include "device_launch_parameters.h" //making threadIdx/blockIdx/blockDim/gridDim visible

int cxxx = 0;
inline ValueType AtomicSet(int v, ValueType weight, ValueType *dis);
//extern Graph *ggg;
string dir = "/home/lhb/compress1/com1/CompressGraph/dataset/cnr-2000/origin";
Graph graph(dir);

const int vert_num = graph.vert_num;
const int edge_num = graph.edge_num;


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
	cudaMallocHost(&distance, sizeof(ValueType)*vert_num);
	cudaMallocHost(&global_ft, sizeof(int)*vert_num);
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
		cudaMallocHost(&distance, sizeof(ValueType)*vert_num);
		cudaMallocHost(&global_ft, sizeof(int)*vert_num);
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
//cout<<"test------ggg:"<<ggg->vert_num<<endl;
//
//return;
	set<Vertex> active_vert(global_ft, global_ft + global_ft_cnt);
	set<Vertex> active_vertx;
cout<<cxxx++<<"	"<<global_ft_cnt<<endl;
	int iteration_id = 0;
	bool flag =true;
	int *iteration_act_num = new int[100];
	while(active_vert.size()>0||active_vertx.size()>0) {
		if(flag) {
			iteration_act_num[iteration_id]=active_vert.size();
			for(auto it = active_vert.begin(); it != active_vert.end(); ++it){
				int u = *it;
				for(int j = graph.csr_v[u]; j < graph.csr_v[u+1]; j++)
				{
					int v = graph.csr_e[j];
					ValueType w = distance[u] + graph.csr_w[j];
					if(w < distance[v]){
						distance[v] = w;
						active_vertx.insert(v);
					}
				}
			}
			
			active_vert.clear();
			iteration_id++;
			flag = false;
		} else {
			iteration_act_num[iteration_id]=active_vertx.size();
			for(auto it = active_vertx.begin(); it != active_vertx.end(); ++it){
				int u = *it;
				for(int j = graph.csr_v[u]; j < graph.csr_v[u+1]; j++)
				{
					int v = graph.csr_e[j];
					ValueType w = distance[u] + graph.csr_w[j];
					if(w < distance[v]){
						distance[v] = w;
						active_vert.insert(v);
					}
				}
			}
			active_vertx.clear();
			iteration_id++;
			flag = true;
		}
	}
	/*for(int i = 0 ;iteration_act_num[i]!=0 ; i++){
		cout <<i<< "	act_num："<<iteration_act_num[i]<<endl;
		if(i>198) break;
	}
	delete[] iteration_act_num;*/
}

//atomic 实现dis更新
inline ValueType AtomicSet(int v, ValueType weight, ValueType *distance){
	if (sizeof(ValueType) == 4) {
		volatile ValueType old_val, new_val;
		do {
			old_val = distance[v];
			new_val = weight;
		} while(!__sync_bool_compare_and_swap((int*)distance+v, *((int*)&old_val), *((int*)&new_val)));
		return old_val;
	} else {
		std::cout << "CAS bad length" << std::endl;
		exit(-1);
	}
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
