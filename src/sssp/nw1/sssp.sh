#!/bin/bash

nvcc -std=c++14 SSSPMain.cu -o result
./result	/home/lhb/compress1/com1/CompressGraph/dataset/cnr-2000/filter
