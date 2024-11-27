exec=../bin/filterx
exec3=../bin/read2binary
exec4=../bin/getruleorder


# 
threshold=(0 1 2 3 4 5 6 7 8 10 12 15 17 20 24 27 30 35 48 63 80 99 120 143 168 200 300 400 500 600 700 1000 2000 3000 5000 7000 10000 20000  50000 70000 100000 200000 400000 600000 800000)

for ((i=0; i<${#threshold[@]}; i++));
do
num="${threshold[i]}"

$exec ../dataset/cnr-2000/compress/csr_vlist.bin ../dataset/cnr-2000/compress/csr_elist.bin ../dataset/cnr-2000/compress/info.bin  $num
size1=$(stat -c "%s" ../dataset/cnr-2000/origin/csr_vlist.bin)
size2=$(stat -c "%s" ../dataset/cnr-2000/origin/csr_elist.bin)
size3=$(stat -c "%s" ../dataset/cnr-2000/filter/csr_vlist.bin)
size4=$(stat -c "%s" ../dataset/cnr-2000/filter/csr_elist.bin)

ratio=$(echo "scale=4; ($size1 + $size2)/ ($size3 + $size4)" | bc)
echo "文件1与文件2的大小比值为: $ratio "
echo ": $num" 


#记录rule_order
$exec4 ../dataset/cnr-2000/filter/csr_vlist.bin ../dataset/cnr-2000/filter/csr_elist.bin ../dataset/cnr-2000/filter/info.bin

#移除分支的优化程序
nvcc -std=c++14 c1/main.cu -o res
./res ../dataset/cnr-2000/filter

rm -f res

done