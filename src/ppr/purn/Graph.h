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

    vector<Vertex> csr_ov;
    vector<Vertex> csr_oe;
    vector<ValueType> csr_ow;


    Vertex vert_num;
    Vertex edge_num;

    // Construct Graph based on the input file (edgelist or adjacency list)
    Graph(string _data_folder) {
        this->data_folder = _data_folder;
        this->vert_num = read_binary2vector(data_folder + "/ppr/vlist.bin", this->csr_v) - 1;
        this->edge_num = read_binary2vector(data_folder + "/ppr/elist.bin", this->csr_e);
        int cnt = read_binary2vector(data_folder + "/ppr/wlist.bin", this->csr_w);

        cnt = read_binary2vector(data_folder + "/ppr/hvlist.bin", this->csr_ov);
        cnt = read_binary2vector(data_folder + "/ppr/helist.bin", this->csr_oe);
        cnt = read_binary2vector(data_folder + "/ppr/hwlist.bin", this->csr_ow);
        cout << "INIT--class Graph is constructed" << endl;
        cout<<"The vertex num is:"<<vert_num<<endl;
        cout<<"The edge num is:"<<edge_num<<endl;
    }

    ~Graph() {
        vector<Vertex>().swap(this->csr_v);
        vector<Vertex>().swap(this->csr_e);

        vector<ValueType>().swap(this->csr_w);

        vector<Vertex>().swap(this->csr_ov);
        vector<Vertex>().swap(this->csr_oe);

        vector<ValueType>().swap(this->csr_ow);

        cout << "CLEAR--class Graph is destroyed" << endl;
    }
};
