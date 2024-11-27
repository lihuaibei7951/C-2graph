exec=../bin/filterz
exec3=../bin/read2binary
exec4=../bin/getruleorder


#threshold=(1 2 3 4 5 6 7 8 10 12 15 17 20 24 27 30 35 48 63 80 99 120 143 168 200 300 400 500 600 700 1000 2000 3000 5000 7000 10000 20000  50000 70000 100000)

threshold=(1000)
level=(0)
#yyy=(49	49	37	37	35	35	31	31	29	29	24	24	22	19	17	16	15	15	13	11	11	10	10	11	11	8	10	10	10	8	6	5	4	4	3	3	2	2	2	1)
#threshold=(32)
#level=(0)
yyy=(1000)
for ((i=0; i<${#threshold[@]}; i++));
do
num="${threshold[i]}"
yy="${yyy[i]}"

for nn in "${level[@]}"
do
#threshold && level

if [ $yy -gt $nn ]; then
$exec ../dataset/cnr-2000/compress/csr_vlist.bin ../dataset/cnr-2000/compress/csr_elist.bin ../dataset/cnr-2000/compress/info.bin  $num  $nn
size1=$(stat -c "%s" ../dataset/cnr-2000/origin/csr_vlist.bin)
size2=$(stat -c "%s" ../dataset/cnr-2000/origin/csr_elist.bin)
size3=$(stat -c "%s" ../dataset/cnr-2000/filter/csr_vlist.bin)
size4=$(stat -c "%s" ../dataset/cnr-2000/filter/csr_elist.bin)

ratio=$(echo "scale=4; ($size1 + $size2)/ ($size3 + $size4)" | bc)
echo "文件1与文件2的大小比值为: $ratio "
echo ": $num" 
echo ": $nn"

#记录degree
$exec3 sss 3

#记录rule_order
$exec4 ../dataset/cnr-2000/filter/csr_vlist.bin ../dataset/cnr-2000/filter/csr_elist.bin ../dataset/cnr-2000/filter/info.bin

#移除分支的优化程序
nvcc -std=c++14 c-cg-0714/SSSPMain.cu -o res
./res ../dataset/cnr-2000
rm -f res

fi

done

done