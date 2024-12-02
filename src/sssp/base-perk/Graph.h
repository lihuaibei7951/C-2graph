#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <sstream>
#include <sys/stat.h>
#include <vector>
#include<math.h>
#include <random>
#include "core/io.hpp"
#include <algorithm> // for std::clamp
using namespace std;

typedef int ValueType;
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
    ValueType rmax;
    Graph(){

    }

    // Construct Graph based on the input file (edgelist or adjacency list)
    void Graphinit(string _data_folder) {
        this->data_folder = _data_folder;
        this->vert_num  = read_binary2vector(data_folder + "/origin/vlist.bin",this->csr_v)-1;
        this->edge_num = read_binary2vector(data_folder + "/origin/elist.bin",this->csr_e);
        int wsize = read_binary2vector(data_folder + "/origin/wlist.bin",this->csr_w);

        this->indegree.resize(vert_num,0);

        for(int i=0;i<vert_num;i++){
            for(int j=csr_v[i];j<csr_v[i+1];j++){
                indegree[csr_e[j]]+=1;
            }
        }

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


