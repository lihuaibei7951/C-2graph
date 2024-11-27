//
// Created by 65766 on 2024/7/13.
//
#include <iostream>
#include <utility>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <limits>
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
#include <numeric>  // std::accumulate
using namespace std;

typedef int Vertex;
typedef int ll;
using ValueType = float;

double alpha=0.8;
// 从文件中读取标签



// 生成长度为 n 且和为 1 的随机标签
vector<ValueType> generateRandomLabel(int n) {
    vector<ValueType> label(n);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    // 生成随机数
    for (int i = 0; i < n; ++i) {
        label[i] = dis(gen);
    }

    // 计算随机数的和
    double sum = accumulate(label.begin(), label.end(), 0.0);

    // 归一化，使得和为 1
    for (int i = 0; i < n; ++i) {
        label[i] /= sum;
        label[i] *= (1 - alpha);
    }

    return label;
}

// 生成多个初始化标签，并展平为一维向量
vector<ValueType> generateInitialLabels(int numLabels, int labelLength) {
    vector<ValueType> initialLabels(numLabels * labelLength);
    for (int i = 0; i < numLabels; ++i) {
        vector<ValueType> label = generateRandomLabel(labelLength);
        copy(label.begin(), label.end(), initialLabels.begin() + i * labelLength);
    }
    return initialLabels;
}
// 从文件中读取标签
std::vector<ValueType> readLabelsFromFile(const std::string& filename, int numLabels, int labelLength) {
    std::vector<ValueType> labels(numLabels * labelLength);
    std::ifstream file(filename);
    if (file.is_open()) {
        for (int i = 0; i < numLabels * labelLength; ++i) {
            file >> labels[i];
        }
        file.close();
    }
    return labels;
}
// 将标签写入文件
void writeLabelsToFile(const std::string& filename, const std::vector<ValueType>& labels) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (double label : labels) {
            file << label << " ";
        }
        file.close();
    }
}
bool file_exists(const std::string& filename);

long file_size(std::string filename);

template<typename VType>
vector<VType> read_binary2vector(VType i,std::string filename);


class Graph {
  public:
    // graph topology stored in adjacency lists:
    // source_id->target_id1:target_id2:...
    string data_folder;
    vector<ll> csr_v;
    vector<Vertex> csr_e;
    vector<Vertex> csr_w;

    vector<bool> active;
    vector<bool> flag;

    vector<Vertex>  weight_sum;

    vector<ValueType> pi;
    vector<ValueType> labels;
    vector<ValueType> new_labels;


    Vertex vert_num;
    Vertex label_num=2;
    ll edge_num;


    // Construct Graph based on the input file (edgelist or adjacency list)
    Graph(string _data_folder) {
        this->data_folder = std::move(_data_folder);
        ll x=1;
        Vertex y=1;
        ValueType z=1.0;
        this->csr_v = read_binary2vector(x,data_folder + "/csr_vlist.bin");
        this->csr_e = read_binary2vector(y,data_folder + "/csr_elist.bin");
        this->csr_w = read_binary2vector(y,data_folder + "/csr_wlist.bin");

        this->vert_num = csr_v.size() - 1;
        this->edge_num = csr_e.size();
        this->active.resize(vert_num,true);
        this->flag.resize(vert_num,false);

        // 初始化总权重,实际计算应该为入度和？
        weight_sum.resize(vert_num,0);
        for (int i=0;i<vert_num;i++) {
            for(int j=csr_v[i];j<csr_v[i+1];j++){
                weight_sum[csr_e[j]]+=csr_w[j];
            }

        }

        if (file_exists(data_folder + "/label.txt")) {
            labels = readLabelsFromFile(data_folder + "/label.txt", vert_num, label_num);
        } else {
            labels = generateInitialLabels(vert_num, label_num);
            writeLabelsToFile(data_folder + "/label.txt", labels);
        }
        pi=labels;
        new_labels.resize(vert_num*label_num,0);


        cout << "INIT--class Graph is constructed" << endl;
        cout<<"The vertex num is:"<<vert_num<<endl;
        cout<<"The edge num is:"<<edge_num<<endl;

    }

    ~Graph() {
        vector<ll>().swap(this->csr_v);
        vector<Vertex>().swap(this->csr_e);
        vector<Vertex>().swap(this->csr_w);

        cout << "CLEAR--class Graph is destroyed" << endl;
    }
};


bool file_exists(const std::string& filename) {
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

