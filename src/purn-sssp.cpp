//
// Created by lihuaibei on 24-9-9.
//
#include <omp.h>
#include <string.h>
#include <unistd.h>
#include <thread>
#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>
#include <core/io.hpp>
#include <core/util.hpp>
#include <fstream>

/*
 * 考虑到顶点之间的互相限制，先写一个只涉及一层的SSSP算法。
 * 测试剪枝的效果。
 * 这个版本基于vector，加入了权重计算过程。
 * 在此基础上对入度=2的节点进行处理
 * 加入了顶点属性标志的flag,u与root,v互斥
 *
 */

template <typename T>
void insert_sorted(std::vector<T>& vec, const T& value) {
    // 使用 std::lower_bound 找到插入位置
    auto it = std::lower_bound(vec.begin(), vec.end(), value);

    // 如果元素不存在，则插入元素
    if (it == vec.end() || *it != value) {
        vec.insert(it, value);
    }
}

/*
 * convert csr to double vector in-edge
 */

void csr_convert_idx(std::vector<std::vector<std::pair<VertexT,bool>>> &graph,
                     std::vector<VertexT> &vlist, std::vector<VertexT> &elist,
                     VertexT vertex_cnt) {
    // #pragma omp parallel for schedule(dynamic)
    for (VertexT v = 0; v < vertex_cnt; v++) {
        VertexT start = vlist[v];
        VertexT end = vlist[v + 1];
        std::set<VertexT> myset;
        for (VertexT e = start; e < end; e++) {
            //graph[elist[e]].push_back(v);
            if(v==elist[e]) continue;
            myset.insert(elist[e]);
        }

        for(auto it = myset.begin(); it != myset.end(); ++it) {
            graph[*it].emplace_back(v,1);
        }
    }
}


int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: filter <vlist> <elsit> <wlist> \n");
        return 0;
    }
    std::string path = argv[1];
    std::string pvlist = path + "/origin/vlist.bin";
    std::string pelist = path + "/origin/elist.bin";
    std::string pwlist = path + "/origin/wlist.bin";

    // 1. 打开二进制文件，并清空内容
    std::ofstream files1(path + "/ppr/hvlist.bin", std::ios::binary | std::ios::trunc);
    std::ofstream files2(path + "/ppr/helist.bin", std::ios::binary | std::ios::trunc);
    std::ofstream files3(path + "/ppr/hwlist.bin", std::ios::binary | std::ios::trunc);
    if (!files1||!files2||!files3) {
        std::cerr << "无法打开文件！" << std::endl;
    }
    files1.close();  // 关闭文件，完成清空操作
    files2.close();  // 关闭文件，完成清空操作
    files3.close();  // 关闭文件，完成清空操作

    // 2. 以追加模式打开文件，依次写入 vector<int> 数据
    std::ofstream file1(path + "/ppr/hvlist.bin", std::ios::binary | std::ios::app);
    std::ofstream file2(path + "/ppr/helist.bin", std::ios::binary | std::ios::app);
    std::ofstream file3(path + "/ppr/hflist.bin", std::ios::binary | std::ios::app);
    if (!file1||!file2||!file3) {
        std::cerr << "无法打开文件！" << std::endl;
    }
    std::vector<VertexT> tmpedge;
    std::vector<ValueT> tmpweight;
    int zero=0;
    file1.write(reinterpret_cast<const char*>(&zero),sizeof(int ));

    std::vector<VertexT> vlist;
    std::vector<VertexT> elist;
    std::vector<VertexT> wlist;

    int v_cnt = read_binary2vector(pvlist, vlist)-1;
    int e_cnt = read_binary2vector(pelist, elist);
    int w_cnt = read_binary2vector(pwlist, wlist);
    int vertex_cnt=v_cnt;

    fprintf(stderr, "purn start...\n");
    printf("init vert : %d\n",vertex_cnt);
    printf("init edge : %d\n",e_cnt);
    double start = timestamp();

    //第一步：获取graph以及graphT
    std::vector<std::vector<std::pair<VertexT,VertexT>>> graph(vertex_cnt);
    csr_convert(graph, vlist, elist, wlist, vertex_cnt);
    wlist.resize(0);
    // 强制释放内存
    std::vector<int>().swap(wlist);
    std::vector<std::vector<std::pair<VertexT,bool>>> graphT(vertex_cnt);
    csr_convert_idx(graphT, vlist, elist, vertex_cnt);
    elist.resize(0);
    // 强制释放内存
    std::vector<int>().swap(elist);


    int e1=0,e2=0;
    for(int i=0;i<vertex_cnt;i++){
        e1+=graph[i].size();
        e2+=graphT[i].size();
        vlist[i]=graphT[i].size();
    }
    if(e1!=e2) printf("e1!=e2\n");
    printf("1st e1 : %d\n",e1);
    printf("1st e2 : %d\n",e2);

    //第三步，改变图拓扑结构，删除indegree=1的冗余边
    std::vector<std::pair<VertexT,VertexT>> tmp;
    std::vector<std::pair<VertexT,VertexT>> tmp1;
    std::vector<std::pair<VertexT,VertexT>> ans;
    std::vector<VertexT> root;
    std::vector<VertexT> dis;
    std::vector<VertexT> id;
    std::vector<VertexT> tmproot;
    int dmax=5;

    for(int i=0;i<vertex_cnt;i++){
        if(i%1000000==0) std::cout<<i<<std::endl;
        //if(i>1000000) break;

        if(graph[i].size()&&vlist[i]>0&& vlist[i]<=dmax){

            root.resize(vlist[i]);
            dis.resize(vlist[i]);
            id.resize(vlist[i]);

            int j=0;
            for(auto it=graphT[i].begin();it!=graphT[i].end();++it){
                if(!it->second) continue;
                root[j] = it->first;
                //查找权重位置
                auto itx = std::lower_bound(graph[root[j]].begin(), graph[root[j]].end(), std::make_pair(i, 99999),
                                            [](const std::pair<VertexT, VertexT>& a, const std::pair<VertexT, VertexT>& b) {
                                                return a.first < b.first;
                                            });
                //check
                if (itx != graph[root[j]].end() && itx->first == i) {
                    dis[j]=itx->second;
                    if(dis[j]==99999){
                        std::cout << " error \n" << std::endl;
                        return 1;
                    }
                } else {
                    std::cout << "something error2 \n" << std::endl;
                    return 1;
                }
                ++j;

            }
            if(j!=vlist[i]) printf("not equal j&&graphTsize\n");
            for(int k=0;k<j;++k) id[k]=0;

            //需要判定root1->v以及root2->v是否存在
            //由于root1，root2的邻居不会少，即递增，可以记录索引
            for(auto jj=graph[i].begin();jj!=graph[i].end();++jj){
                int v=jj->first;
                int d3=jj->second;
                int cnt=0;
                for(int k=0;k<j;++k){
                    while(id[k]<graph[root[k]].size()-1&&graph[root[k]][id[k]].first<v){
                        id[k]++;
                    }
                    if(graph[root[k]][id[k]].first==v)
                        ++cnt;
                    else
                        tmproot.emplace_back(root[k]);

                }
                if(cnt+2>j){
                    tmp1.push_back({v,d3});
                    for (int k : tmproot) {
                        if(k==v)
                            continue;
                        auto it=std::lower_bound(graphT[v].begin(),graphT[v].end(),k,[](const std::pair<VertexT,bool>& a,const VertexT& b){
                            return a.first < b;
                        });
                        if(it==graphT[v].end()||it->first!=k){
                            graphT[v].insert(it, std::make_pair(k,1));
                            vlist[v]++;
                        }
                        else if(it != graphT[v].end()&&it->first==k&&!it->second){
                            vlist[v]++;
                            it->second=true;
                        }

                    }
                    // 删除v中的i->v

                    auto it=std::lower_bound(graphT[v].begin(),graphT[v].end(),i,[](const std::pair<VertexT,bool>& a,const VertexT& b){
                        return a.first < b;
                    });
                    vlist[v]--;
                    it->second= false;

                }else {
                    tmp.push_back({v,d3});
                }
                tmproot.resize(0);
            }
            for(int k=0;k<j;k++){
                int ii=0,jj=0;
                ans.resize(0);
                while(ii<graph[root[k]].size()||jj<tmp1.size()){
                    if(ii==graph[root[k]].size()){
                        if(tmp1[jj].first==root[k]){
                            ++jj;
                            continue;
                        }
                        ans.push_back({tmp1[jj].first,tmp1[jj].second+dis[k]});
                        ++jj;
                        continue;
                    }

                    if(jj==tmp1.size()){
                        ans.push_back(graph[root[k]][ii++]);
                        continue;
                    }

                    if(graph[root[k]][ii].first==tmp1[jj].first){
                        ans.push_back({graph[root[k]][ii].first,std::min(graph[root[k]][ii].second,tmp1[jj].second+dis[k])});
                        ++ii;
                        ++jj;
                    }else if(graph[root[k]][ii].first>tmp1[jj].first){
                        if(tmp1[jj].first==root[k]){
                            ++jj;
                            continue;
                        }
                        ans.push_back({tmp1[jj].first,tmp1[jj].second+dis[k]});
                        ++jj;
                    }else{
                        ans.push_back(graph[root[k]][ii++]);
                    }
                }
                ans.shrink_to_fit(); // 释放多余的内存
                ans.swap(graph[root[k]]);
            }
            tmp.swap(graph[i]);
            tmp.resize(0);
            for(auto it:tmp1){
                tmpedge.emplace_back(it.first);
                tmpweight.emplace_back(it.second);
            }
            zero+=tmp1.size();
            file1.write(reinterpret_cast<const char*>(&zero),sizeof(int));
            file2.write(reinterpret_cast<const char*>(tmpedge.data()), tmpedge.size() * sizeof(int));
            file3.write(reinterpret_cast<const char*>(tmpweight.data()), tmpweight.size() * sizeof(int));
            tmp1.resize(0);
            tmpedge.resize(0);
            tmpweight.resize(0);
        }


    }

    file1.close();  // 关闭文件，完成清空操作
    file2.close();  // 关闭文件，完成清空操作
    file3.close();  // 关闭文件，完成清空操作
    //检查一下
    e1=0,e2=0;
    int edge_num=0;
    for(int i=0;i<vertex_cnt;i++){
        edge_num+=graph[i].size();
        e2+=vlist[i];

    }
    graphT.resize(0);
    std::vector<std::vector<std::pair<VertexT,bool>>>().swap(graphT);


    elist.resize(edge_num);
    wlist.resize(edge_num);
    vlist[0]=0;


    for(int i=0;i<vertex_cnt;i++){
        for(auto & j : graph[i]){
            elist[e1]=std::get<0>(j);
            wlist[e1++]=std::get<1>(j);
        }
        vlist[i+1]=e1;

    }
    elist.resize(e1);
    wlist.resize(e1);
    if(e1!=e2) printf("e1!=e2\n");
    printf("2st e1 : %d\n",e1);
    printf("2st e2 : %d\n",e2);
    printf("the added gr size is : %d\n",zero);


    double end = timestamp();
    fprintf(stderr, "Filter time : %.4f(s)\n", end - start);

    FILE *fvlist;
    FILE *felist;
    FILE *fwlist;
    fvlist = fopen((path + "/sssp/vlist.bin").c_str(), "w");
    felist = fopen((path + "/sssp/elist.bin").c_str(), "w");
    fwlist = fopen((path + "/sssp/wlist.bin").c_str(), "w");
    fwrite(&vlist[0], sizeof(int), vlist.size(), fvlist);


    fwrite(&elist[0], sizeof(int), elist.size(), felist);
    fwrite(&wlist[0], sizeof(int), wlist.size(), fwlist);
    fclose(fvlist);
    fclose(felist);
    fclose(fwlist);

    return 0;
}
