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

