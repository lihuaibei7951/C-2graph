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

vector<Vertex> read_binary2vector(std::string filename);
class Graph {
  public:
    // graph topology stored in adjacency lists:
    // source_id->target_id1:target_id2:...
    string data_folder;
    vector<Vertex> csr_v;
    vector<Vertex> csr_e;
    vector<ValueType> csr_w;

    Vertex vert_num; 
    Vertex edge_num; 

    // Construct Graph based on the input file (edgelist or adjacency list)
    Graph(string _data_folder) {
        this->data_folder = _data_folder;

        this->csr_v = read_binary2vector(data_folder + "/csr_vlist.bin");
        this->csr_e = read_binary2vector(data_folder + "/csr_elist.bin");
        vector<Vertex> info = read_binary2vector(data_folder + "/info.bin");
        
        this->vert_num = csr_v.size()-1;
        this->edge_num = csr_e.size();
        this->csr_w.resize(csr_e.size());
        for(Vertex i = 0;i<csr_e.size();i++){
        		if(csr_e[i]<info[0]){
        			csr_w[i]=1;
        		} else {
        			csr_w[i]=0;
        		}
        } 
	     

        cout << "INIT--class Graph is constructed" << endl;
        cout<<"The vertex num is:"<<vert_num<<endl;
        cout<<"The rule num is:"<<edge_num<<endl;
        
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