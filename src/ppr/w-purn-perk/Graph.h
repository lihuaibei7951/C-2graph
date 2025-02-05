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

inline bool file_exists(std::string filename) {
    struct stat st;
    return stat(filename.c_str(), &st) == 0;
}

inline long file_size(std::string filename) {
    struct stat st;
    assert(stat(filename.c_str(), &st) == 0);
    return st.st_size;
}

template<typename Valtype>
inline vector<Valtype> read_binary2vector(Valtype x, std::string filename) {
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

class Graph {
public:
    // graph topology stored in adjacency lists:
    // source_id->target_id1:target_id2:...
    string data_folder;
    vector<Vertex> csr_v;
    vector<Vertex> csr_e;
    vector<ValueType> csr_w;
    vector<Vertex> indegree;


    vector<Vertex> csr_ov;
    vector<Vertex> csr_oe;
    vector<ValueType> csr_ow;

    Vertex vert_num;
    Vertex edge_num;
    ValueType rmax;
    Graph(){

    }

    // Construct Graph based on the input file (edgelist or adjacency list)
    void Graphinit(string _data_folder) {
        this->data_folder = _data_folder;
        int x=1;
        ValueType y=1.0;

        this->csr_v = read_binary2vector(x,data_folder + "/origin/vlist.bin");
        this->csr_e = read_binary2vector(x,data_folder + "/origin/elist.bin");
        this->vert_num = csr_v.size()-1;
        this->edge_num = csr_e.size();
        this->csr_w = read_binary2vector(y,data_folder + "/origin/flist.bin");

        //this->csr_ov = read_binary2vector(x,data_folder + "/ppr/hvlist.bin");
        //this->csr_oe = read_binary2vector(x,data_folder + "/ppr/helist.bin");
        //this->csr_ow = read_binary2vector(y,data_folder + "/ppr/hwlist.bin");
        this->csr_ov = read_binary2vector(x,data_folder + "/origin/vlist.bin");
        this->csr_oe = read_binary2vector(x,data_folder + "/origin/elist.bin");
        this->csr_ow = read_binary2vector(y,data_folder + "/origin/flist.bin");

        this->indegree.resize(vert_num,0);

        for(int i=0;i<vert_num;i++){
            for(int j=csr_v[i];j<csr_v[i+1];j++){
                indegree[csr_e[j]]+=1;
            }
        }
        this->rmax = 0.001f/vert_num;
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
