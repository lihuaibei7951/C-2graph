#!/bin/bash

nvcc -std=c++14  PPRMain.cu -o res
./res ../../dataset/cnr-2000/origin

#./res	/home/lhb/cucode/CGgraphV1/graph/graph/cnr2000
