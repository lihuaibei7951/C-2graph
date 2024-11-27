#!/bin/bash


#移除分支的优化
nvcc -std=c++14 main.cu -o res
./res ../../dataset/cnr-2000/filter
