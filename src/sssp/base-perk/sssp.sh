#!/bin/bash

nvcc -c -std=c++14 SSSPMain.cu -lpthread
g++ -c -std=c++14 SSSP.cpp
g++ -o  result SSSP.o SSSPMain.o  -lcudart
rm SSSP.o SSSPMain.o 
