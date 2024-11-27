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
     Graph(){}
    // Construct Graph based on the input file (edgelist or adjacency list)
    Graph(string _data_folder) {
        this->data_folder = _data_folder;

        this->csr_v = read_binary2vector(data_folder + "/csr_vlist.bin");
        this->csr_e = read_binary2vector(data_folder + "/csr_elist.bin");
        this->vert_num = csr_v.size()-1;
        this->edge_num = csr_e.size();
        this->csr_w.resize(csr_e.size(),1);   
	     

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
