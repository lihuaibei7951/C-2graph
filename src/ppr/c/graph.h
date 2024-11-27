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
    vector<Vertex> degree;
    vector<Vertex> order;
    vector<Vertex> info;

    Vertex vert_num, rule_num; // #vertices (id <= 4billion)
    Vertex edge_num,vaddr,origin_edge; // #vertices (id <= 4billion)
    Vertex max_step;

    // Construct Graph based on the input file (edgelist or adjacency list)
    Graph(string _data_folder) {
        this->data_folder = _data_folder;

        this->csr_v = read_binary2vector(data_folder + "/csr_vlist.bin");
        this->csr_e = read_binary2vector(data_folder + "/csr_elist.bin");
        this->degree = read_binary2vector(data_folder + "/degree.bin");

        

        
        this->info = read_binary2vector(data_folder + "/info.bin");
        //cout<<"xxxxx"<<csr_e[0]<<"--"<<csr_e[1]<<"--"<<csr_e[2]<<"--"<<csr_e[3]<<endl;
        //cout<<"yyyyy"<<csr_e[861499]<<"--"<<csr_e[861500]<<endl;
        //cout<<"order:0"<<order[0]<<endl;

        this->vert_num = info[0];
        this->rule_num = info[1];
        
        this->origin_edge = 0;
	   for(Vertex i = 0;i<degree.size();i++){
	   		this->origin_edge += degree[i];
	   }
	   
	   this->edge_num = csr_e.size();
	   this->vaddr = vert_num + rule_num;

	   


        vector<Vertex> ord = read_binary2vector(data_folder + "/order.bin");
	   max_step = 0;
	   for(Vertex i = 0;i<ord.size();i++){
	   		max_step = max(max_step,ord[i]);
	   }
	   
	   cout<<max_step<<endl;
	           
        order.resize(vert_num + rule_num,0);
        
        for(Vertex i = 0;i<rule_num;i++){
        	order[vert_num + i]=ord[i];
        }
        

        cout << "INIT--class Graph is constructed" << endl;
        cout<<"The vertex num is:"<<vert_num<<endl;
        cout<<"The rule num is:"<<rule_num<<endl;
        cout<<"test:"<<csr_v.size();
        
    }

    ~Graph() {
        vector<Vertex>().swap(this->csr_v);
        vector<Vertex>().swap(this->csr_e);
        vector<Vertex>().swap(this->degree);
        vector<Vertex>().swap(this->order);
        vector<Vertex>().swap(this->info);

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