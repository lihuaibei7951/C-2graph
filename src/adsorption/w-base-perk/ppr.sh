#!/bin/bash

nvcc -c -std=c++14 PPRMain.cu -lpthread
g++ -c -std=c++14 PPR.cpp 
g++ -o result PPRMain.o PPR.o -lcudart -lpthread
#g++ -o result PPRMain.o -lcudart -lpthread
rm PPRMain.o PPR.o
./result ../../dataset/cnr-2000/origin
