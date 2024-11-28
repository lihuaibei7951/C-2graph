#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <sstream>
#include <sys/stat.h>
#include <vector>
#include<math.h>
#include <core/io.hpp>
using namespace std;

typedef float ValueType;
typedef int Vertex;

class Graph {
public:
    // graph topology stored in adjacency lists:
    // source_id->target_id1:target_id2:...
    string data_folder;
    vector<Vertex> csr_v;
    vector<Vertex> csr_e;
    vector<ValueType> csr_w;

    vector<Vertex> indegree;

    Vertex vert_num;
    Vertex edge_num;

    // Construct Graph based on the input file (edgelist or adjacency list)
    Graph(string _data_folder) {
        this->data_folder = _data_folder;

        this->vert_num  = read_binary2vector(data_folder + "/origin/vlist.bin",this->csr_v)-1;
        this->edge_num = read_binary2vector(data_folder + "/origin/elist.bin",this->csr_e);
        int wsize = read_binary2vector(data_folder + "/origin/flist.bin",this->csr_w);


        cout << "INIT--class Graph is constructed" << endl;
        cout<<"The vertex num is:"<<vert_num<<endl;
        cout<<"The edge num is:"<<edge_num<<endl;

    }

    ~Graph() {
        vector<Vertex>().swap(this->csr_v);
        vector<Vertex>().swap(this->csr_e);
        vector<ValueType>().swap(this->csr_w);

        cout << "CLEAR--class Graph is destroyed" << endl;
    }
};