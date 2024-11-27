#!/bin/bash

nvcc -c -std=c++14 SSSPMain.cu -lpthread
g++ -c -std=c++14 SSSP.cpp -fcilkplus -lcilkrts
g++ -o  result SSSP.o SSSPMain.o  -lcudart -fcilkplus

rm SSSP.o SSSPMain.o 
./result	/home/lhb/compress1/com1/CompressGraph/dataset/cnr-2000/origin
