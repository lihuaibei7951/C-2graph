#pragma once
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <set>
#include <utility>//std:pair
#define VertexT int
#define VertexL long long
#define ValueT  float
#define INIT 100000
#define INF 1e-6
#define OMP_THRESHOLD 1024

/*
 *  convert csr to double vector
 */

double timestamp() {
    timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + 1e-6 * t.tv_usec;
}

/*
 *  convert csr to double vector out-edge
 */
void csr_convert(std::vector<std::vector<VertexT>> &graph,
                 std::vector<VertexT> &vlist, std::vector<VertexT> &elist,
                 VertexT vertex_cnt) {
    // #pragma omp parallel for schedule(dynamic)
    for (VertexT v = 0; v < vertex_cnt; v++) {
        VertexT start = vlist[v];
        VertexT end = vlist[v + 1];
        std::set<VertexT> myset;
        for (VertexT e = start; e < end; e++) {
            //graph[v].push_back(elist[e]);
            if(v==elist[e]) continue;
            myset.insert(elist[e]);
        }
        for(auto it = myset.begin(); it != myset.end(); ++it) {
			graph[v].push_back(*it);
        }
    }
}

/*
 * convert csr to double vector in-edge
 */

void csr_convert_idx(std::vector<std::vector<VertexT>> &graph,
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
			graph[*it].push_back(v);
        }
    }
}

/*
 *  weight  for out-degree
 */
void csr_convert(std::vector<std::vector<std::pair<VertexT,VertexT>>> &graph,
                 std::vector<VertexT> &vlist, std::vector<VertexT> &elist,std::vector<VertexT> &wlist,
                 VertexT vertex_cnt) {
    // #pragma omp parallel for schedule(dynamic)
    for (VertexT v = 0; v < vertex_cnt; v++) {
        VertexT start = vlist[v];
        VertexT end = vlist[v + 1];
        std::set<std::pair<VertexT,VertexT>> myset;
        for (VertexT e = start; e < end; e++) {
            if(v==elist[e]) continue;
            myset.insert({elist[e],wlist[e]});
        }
        for(auto it = myset.begin(); it != myset.end(); ++it) {
			graph[v].emplace_back(it->first,it->second);
        }
    }
}

void csr_convert(std::vector<std::vector<std::pair<VertexT,ValueT>>> &graph,
                 std::vector<VertexT> &vlist, std::vector<VertexT> &elist,std::vector<ValueT> &wlist,
                 VertexT vertex_cnt) {
    // #pragma omp parallel for schedule(dynamic)
    for (VertexT v = 0; v < vertex_cnt; v++) {
        VertexT start = vlist[v];
        VertexT end = vlist[v + 1];
        std::set<std::pair<VertexT,ValueT>> myset;
        for (VertexT e = start; e < end; e++) {
            //graph[v].push_back(elist[e]);
            if(v==elist[e]) continue;
            myset.insert({elist[e],wlist[e]});
        }
        for(auto it = myset.begin(); it != myset.end(); ++it) {
            graph[v].push_back({it->first,it->second});
        }
    }
}

//重写数据
/*
 *  convert csr to double vector out-edge
 */
void csr_convert(std::vector<std::set<VertexT>> &graph,
                 std::vector<VertexT> &vlist, std::vector<VertexT> &elist,
                 VertexT vertex_cnt) {
    // #pragma omp parallel for schedule(dynamic)
    for (VertexT v = 0; v < vertex_cnt; v++) {
        VertexT start = vlist[v];
        VertexT end = vlist[v + 1];
        for (VertexT e = start; e < end; e++) {
            if(v==elist[e]) continue;
            graph[v].insert(elist[e]);
        }
    }
}

/*
 * convert csr to double vector in-edge
 */

void csr_convert_idx(std::vector<std::set<VertexT>> &graph,
                     std::vector<VertexT> &vlist, std::vector<VertexT> &elist,
                     VertexT vertex_cnt) {
    // #pragma omp parallel for schedule(dynamic)
    for (VertexT v = 0; v < vertex_cnt; v++) {
        VertexT start = vlist[v];
        VertexT end = vlist[v + 1];
        for (VertexT e = start; e < end; e++) { 
            if(v==elist[e]) continue;
            graph[elist[e]].insert(v);
        }
    }
}


/*
 *  convert csr to double vector out-edge && rule_graph
 */
void csr_convert_rule(std::vector<std::vector<VertexT>> &graph, std::vector<std::vector<VertexT>> &rule,
                 std::vector<VertexT> &vlist, std::vector<VertexT> &elist,
                 VertexT v_cnt, VertexT vertex_cnt) {
    // #pragma omp parallel for schedule(dynamic)
    for (VertexT v = 0; v < vertex_cnt; v++) {
        VertexT start = vlist[v];
        VertexT end = vlist[v + 1];
        for (VertexT e = start; e < end; e++) {
            graph[v].push_back(elist[e]);
            if(v>=v_cnt&&elist[e]>=v_cnt){
            	rule[v-v_cnt].push_back(elist[e]-v_cnt);
            	if(v<elist[e]) std::cout<<"error1"<<std::endl;
            }
        }
        
    }
}

/*
 * convert csr to double vector in-edge && rule_graphT
 */

void csr_convert_idx_rule(std::vector<std::vector<VertexT>> &graph,std::vector<std::vector<VertexT>> &rule,
                     std::vector<VertexT> &vlist, std::vector<VertexT> &elist,
                     VertexT v_cnt,VertexT vertex_cnt) {
    // #pragma omp parallel for schedule(dynamic)
    for (VertexT v = 0; v < vertex_cnt; v++) {
        VertexT start = vlist[v];
        VertexT end = vlist[v + 1];
        for (VertexT e = start; e < end; e++) {  
            graph[elist[e]].push_back(v);
            if(v>=v_cnt&&elist[e]>=v_cnt){
            	rule[elist[e]-v_cnt].push_back(v-v_cnt);
            	if(v<elist[e]) std::cout<<"error2"<<std::endl;
            }
        }
    }
}


void insert_spliter(std::vector<int> &vlist, std::vector<int> &elist,
                    std::vector<int> &re, int max_symbol) {
    int v, e;
    int start, end;
    if (vlist.size() == 0)
        return;
    int cur = max_symbol;
    for (v = 0; v < vlist.size() - 1; v++) {
        start = vlist[v];
        end = vlist[v + 1];
        for (e = start; e < end; e++) {
            re.push_back(elist[e]);
        }
        re.push_back(cur++);
    }
}

void genNewIdForRule(std::vector<VertexT> &newRuleId, VertexT &newRule_cnt,
                     std::vector<bool> &merge_flag, VertexT vertex_cnt,
                     VertexT rule_cnt) {
    VertexT cur = vertex_cnt;
    for (VertexT i = 0; i < rule_cnt; i++) {
        newRuleId[i] = cur;
        if (merge_flag[i] == false) {
            cur++;
        }
    }
    newRule_cnt = cur - vertex_cnt; // now rule count
}

void genNewGraphCSR(std::vector<VertexT> &vlist, std::vector<VertexT> &elist,
                    std::vector<std::vector<VertexT>> &graph,
                    std::vector<VertexT> &newRuleId,
                    std::vector<bool> &merge_flag, VertexT vertex_cnt) {
    VertexT g_size = graph.size();
    vlist.push_back(0);
    VertexT e_size = 0;
    for (VertexT v = 0; v < g_size; ++v) {
        VertexT srcID = v;
        VertexT e_size = graph[srcID].size();
        if (srcID >= vertex_cnt && merge_flag[srcID - vertex_cnt] == true)
            continue;
        for (VertexT e = 0; e < e_size; e++) {
            VertexT dstID = graph[srcID][e];
            VertexT n_dstID =
                dstID >= vertex_cnt ? newRuleId[dstID - vertex_cnt] : dstID;
            elist.push_back(n_dstID);
        }
        e_size = elist.size();
        vlist.push_back(e_size);
    }
}
