#include <iostream>
#include <queue>
#include <algorithm>
#include <time.h>
#include <pthread.h>
#include <vector>
using namespace std;

typedef float ValueType;
typedef int Vertex;

extern Graph graph;
extern const Vertex vert_num;
extern const Vertex edge_num;

extern const ValueType alpha;
extern const ValueType rmax;


//queue struct
typedef struct queuenode{
	ValueType *reserve;
	ValueType *residue;
	int *global_ft;
	int *flag;
	int source;
	int *global_ft_cnt;
	struct queuenode *next;
}Data;

typedef struct
{
	//Data *data;
	int flag;
	int length;
	Data *front, *rear;

}BufferQueue, *PBufferQueue;

//insert queuenode
void insertQueue(BufferQueue *buffer, Data *node);

//del queuenode
int outQueue(BufferQueue *buffer);

//初始化循环队列
void initQueue(BufferQueue *buffer, int MaxSize);

void* ppr_CPU(void* data);




