#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <sstream>
#include <sys/stat.h>
#include <vector>
#include<math.h>
using namespace std;

typedef float ValueType;
typedef int Vertex;

bool file_exists(std::string filename);

long file_size(std::string filename);
template<typename Valtype>
vector<Valtype> read_binary2vector(Valtype x,std::string filename);
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
        int x=1;
        ValueType y=1.0;

        this->csr_v = read_binary2vector(x,data_folder + "/origin/vlist.bin");
        this->csr_e = read_binary2vector(x,data_folder + "/origin/elist.bin");
        this->vert_num = csr_v.size()-1;
        this->edge_num = csr_e.size();
        this->csr_w = read_binary2vector(y,data_folder + "/origin/flist.bin");

        this->indegree.resize(vert_num,0);

        for(int i=0;i<vert_num;i++){
            for(int j=csr_v[i];j<csr_v[i+1];j++){
                indegree[csr_e[j]]+=1;
            }
        }



        cout << "INIT--class Graph is constructed" << endl;
        cout<<"The vertex num is:"<<vert_num<<endl;
        cout<<"The edge num is:"<<edge_num<<endl;
        //cout<<"test:"<<csr_v.size();

    }

    ~Graph() {
        vector<Vertex>().swap(this->csr_v);
        vector<Vertex>().swap(this->csr_e);
        vector<ValueType>().swap(this->csr_w);

        cout << "CLEAR--class Graph is destroyed" << endl;
    }
};
bool file_exists(std::string filename) {
    struct stat st;
    return stat(filename.c_str(), &st) == 0;
}

long file_size(std::string filename) {
    struct stat st;
    assert(stat(filename.c_str(), &st) == 0);
    return st.st_size;
}

template<typename Valtype>
vector<Valtype> read_binary2vector(Valtype x, std::string filename) {
    std::vector<Valtype> out;
    if (!file_exists(filename)) {
        fprintf(stderr, "file:%s not exist.\n", filename.c_str());
        exit(0);
    }
    long fs = file_size(filename);
    long ele_cnt = fs / sizeof(Valtype);
    out.resize(ele_cnt);
    FILE *fp;
    fp = fopen(filename.c_str(), "r");
    if (fread(&out[0], sizeof(Valtype), ele_cnt, fp) < ele_cnt) {
        fprintf(stderr, "Read failed.\n");
    }
    return out;
}