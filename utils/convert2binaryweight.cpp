#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <ctime>

using namespace std;

void buildGraph(const string& filePath, vector<vector<int>>& G) {
    int VerNum = 0;
    cout<<"Reading graph from " << filePath << endl;

    // 第一遍扫描，确定最大顶点标号
    ifstream file(filePath);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filePath << endl;
        return;
    }

    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        int src;
        string dstPart;

        if (!(iss >> src >> dstPart)) {
            continue; // 跳过格式不合法的行
        }

        VerNum = max(VerNum, src); // 更新最大顶点编号

        stringstream dstStream(dstPart);
        string dstStr;
        while (getline(dstStream, dstStr, ':')) {
            int dst = stoi(dstStr);
            VerNum = max(VerNum, dst); // 更新最大顶点编号
        }
    }

    file.close();
    cout<<"Max Vertex Number: " << VerNum << endl;

    // 创建二维 vector
    G.resize(VerNum + 1); // 顶点编号范围 [0, VerNum]

    // 第二遍扫描，填充 G
    file.open(filePath);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filePath << endl;
        return;
    }

    while (getline(file, line)) {
        istringstream iss(line);
        int src;
        string dstPart;

        if (!(iss >> src >> dstPart)) {
            continue;
        }

        stringstream dstStream(dstPart);
        string dstStr;
        while (getline(dstStream, dstStr, ':')) {
            int dst = stoi(dstStr);
            if (src != dst) { // 去除自环
                G[src].push_back(dst);
            }
        }
    }

    file.close();

    // 去重并排序每个顶点的邻接列表
    for (auto& neighbors : G) {
        sort(neighbors.begin(), neighbors.end());
        neighbors.erase(unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
}

int main(int argc, char **argv){
//读取输入属性
    std::string filePath = argv[1];
    std::string path = argv[2];
    vector<vector<int>> G;
    // 构建图
    buildGraph(filePath, G);
    unsigned int VerNum = G.size();
    unsigned int EdgeNum = 0;

    // 输出结果

    for (const auto & i : G) {
        EdgeNum+=i.size();
    }

    std::vector<int> vlist(VerNum+1);
    std::vector<int> elist(EdgeNum);
    std::vector<int> wlist(EdgeNum);
    std::vector<float> flist(EdgeNum);
    vlist[0]=0;
    for(int i=0;i<VerNum;i++){
        vlist[i+1]=vlist[i]+(int)G[i].size();
        int sum=0;
        for(int j=vlist[i];j<vlist[i+1];j++){
            elist[j]=G[i][j-vlist[i]];
            wlist[j]=(rand() % 100)+1 ;
            sum+=wlist[j];
        }
        for(int j=vlist[i];j<vlist[i+1];j++){
            flist[j]=1.0*wlist[j]/sum;
        }
    }

    FILE *fvlist;
    FILE *felist;
    FILE *fwlist;
    FILE *fflist;
    fvlist = fopen((path + "/origin/vlist.bin").c_str(), "w");
    felist = fopen((path + "/origin/elist.bin").c_str(), "w");
    fwlist = fopen((path + "/origin/wlist.bin").c_str(), "w");
    fflist = fopen((path + "/origin/flist.bin").c_str(), "w");
    fwrite(&vlist[0], sizeof(int), vlist.size(), fvlist);
    fwrite(&elist[0], sizeof(int), elist.size(), felist);
    fwrite(&wlist[0], sizeof(int), wlist.size(), fwlist);
    fwrite(&flist[0], sizeof(float), flist.size(), fflist);

    fclose(fvlist);
    fclose(felist);
    fclose(fwlist);
    fclose(fflist);

    return 0;
}
