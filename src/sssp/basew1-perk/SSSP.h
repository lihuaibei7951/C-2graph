#include <iostream>
#include <queue>
#include <algorithm>
#include <time.h>
#include <pthread.h>


typedef float ValueType;
typedef int Vertex;
//extern Graph *ggg;
extern Graph graph;
extern const int vert_num;
extern const int edge_num;

typedef struct queuenode {
	ValueType *distance;
	int *global_ft;
	int *flag;
	int source;
	int *global_ft_cnt;
	struct queuenode *next;

}Data;

typedef struct
{
	int flag;
	int length;	
	Data *front, *rear;

}BufferQueue, *PBufferQueue;

void insertQueue(BufferQueue *buffer, Data *node);

int outQueue(BufferQueue *buffer);

void initQueue(BufferQueue *buffer, int MaxSize);

void* SSSP_CPU(void* data);