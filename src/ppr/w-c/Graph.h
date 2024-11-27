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

template<typename VType>
vector<VType> read_binary2vector(VType i,std::string filename);
class Graph {
  public:
    // graph topology stored in adjacency lists:
    // source_id->target_id1:target_id2:...
    string data_folder;
    vector<Vertex> csr_v;
    vector<Vertex> csr_ov;
    vector<Vertex> csr_e;
    vector<Vertex> csr_idx;
    vector<ValueType> csr_w;

    Vertex vert_num,add_num,rule_num;
    Vertex edge_num,w_num; 

    // Construct Graph based on the input file (edgelist or adjacency list)
    Graph(string _data_folder) {
        this->data_folder = _data_folder;
		int x=1;
		ValueType y=1;
        this->csr_v = read_binary2vector(x,data_folder + "/filter/csr_vlist.bin");
        this->csr_ov = read_binary2vector(x,data_folder + "/origin/vlist.bin");
        this->csr_e = read_binary2vector(x,data_folder + "/filter/csr_elist.bin");
        this->csr_idx = read_binary2vector(x,data_folder + "/filter/csr_idx.bin");
        this->csr_w = read_binary2vector(y,data_folder + "/origin/flist.bin");

        cout<<csr_ov[0]<<" "<<csr_ov[1]<<csr_ov[2]<<" "<<csr_ov[3]<<csr_ov[4]<<" "<<csr_ov[5]<<endl;
        cout<<csr_v[0]<<" "<<csr_v[1]<<csr_v[2]<<" "<<csr_v[3]<<csr_v[4]<<" "<<csr_v[5]<<endl;
        cout<<csr_idx[0]<<" "<<csr_idx[1]<<csr_idx[2]<<" "<<csr_idx[3]<<csr_idx[4]<<" "<<csr_idx[5]<<endl;
        cout<<csr_w[0]<<" "<<csr_w[1]<<csr_w[2]<<" "<<csr_w[3]<<csr_w[4]<<" "<<csr_w[5]<<endl;  
        cout<<csr_e[0]<<" "<<csr_e[1]<<csr_e[2]<<" "<<csr_e[3]<<csr_e[4]<<" "<<csr_e[5]<<endl;  
	    cout<<csr_e[84]<<" "<<csr_w[84]<<endl;

        this->vert_num = csr_ov.size()-1;
        this->add_num = csr_v.size()-1;
        this->rule_num = add_num-vert_num;
        this->edge_num = csr_e.size();
        this->w_num = csr_w.size();
        //this->csr_w.resize(csr_e.size(),1);
        

        cout << "INIT--class Graph is constructed" << endl;
        cout<<"The vertex num is:"<<vert_num<<endl;
        cout<<"The rule num is:"<<rule_num<<endl;
        cout<<"The edge num is:"<<edge_num<<endl;
        
    }

    ~Graph() {
        vector<Vertex>().swap(this->csr_v);
        vector<Vertex>().swap(this->csr_ov);
        vector<Vertex>().swap(this->csr_e);
        vector<Vertex>().swap(this->csr_idx);
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
template<typename VType>
vector<VType> read_binary2vector(VType i,std::string filename) {
    std::vector<VType> out;
    if (!file_exists(filename)) {
        fprintf(stderr, "file:%s not exist.\n", filename.c_str());
        exit(0);
    }
    long fs = file_size(filename);
    long ele_cnt = fs / sizeof(VType);
    out.resize(ele_cnt);
    FILE *fp;
    fp = fopen(filename.c_str(), "r");
    if (fread(&out[0], sizeof(VType), ele_cnt, fp) < ele_cnt) {
        fprintf(stderr, "Read failed.\n");
    }
    return out;
}