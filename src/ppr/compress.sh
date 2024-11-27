#邻接表->二进制
#记录degree
#sh /home/lhb/compress1/com1/CompressGraph/utils/convert_binary.sh

exec1=../bin/convert2binary
exec2=../bin/compress
exec3=../bin/read2binary

#$exec1 	/home/lhb/cucode/Graph/attribute_cnr2000.txt		/home/lhb/cucode/Graph/cnr-2000-hc-termite-pr

#$exec1 	/home/lhb/cucode/Graph/attribute_eu2015.txt		/home/lhb/cucode/Graph/eu-2015-tpd-hc-termite-new

#$exec1	/home/lhb/cucode/Graph/attribute_uk2002.txt		/home/lhb/cucode/Graph/uk-2002-hc-termite-new

#$exec1	/home/lhb/cucode/Graph/attribute_indoc.txt	/home/lhb/cucode/Graph/indochina-2004-hc-termite-new

#$exec1	/home/lhb/cucode/Graph/attribute_livej.txt	/home/lhb/cucode/Graph/livej-new

#$exec1	/home/lhb/cucode/Graph/attribute_orkut.txt	/home/lhb/cucode/Graph/orkut-new

#$exec1	/home/lhb/cucode/Graph/attribute_sk2005.txt	/home/lhb/cucode/Graph/sk-2005-hc-termite-new

#$exec1	/home/lhb/cucode/Graph/attribute_stanford.txt	/home/lhb/cucode/Graph/stanford-new

#$exec1	/home/lhb/cucode/Graph/attribute_arabic-2005-l-hc.txt	/home/lhb/cucode/Graph/arabic-2005-l-hc

#$exec1	/home/lhb/cucode/Graph/attribute_wiki.txt	/home/lhb/cucode/Graph/wiki-new

#$exec1	/home/lhb/cucode/Graph/attribute_it-2004-hc-l-hc.txt	/home/lhb/cucode/Graph/it-2004-hc-l-hc

#$exec1	/home/lhb/cucode/Graph/attribute_gsh-2015-tpd-l-hc.txt	/home/lhb/cucode/Graph/gsh-2015-tpd-l-hc

$exec1   /home/lhb/cucode/Graph/attribute_uk-2007-05@100000-l-hc.txt /home/lhb/cucode/Graph/uk-2007-05@100000-l-hc


#压缩
$exec2 ../dataset/cnr-2000/origin/csr_vlist.bin ../dataset/cnr-2000/origin/csr_elist.bin 2147483647 1	0
#$exec2 ../dataset/cnr-2000/origin/csr_vlist.bin ../dataset/cnr-2000/origin/csr_elist.bin 30000000 3


#filter
exec=../bin/filter
$exec ../dataset/cnr-2000/compress/csr_vlist.bin ../dataset/cnr-2000/compress/csr_elist.bin ../dataset/cnr-2000/compress/info.bin 1

#记录degree
#$exec3 sss -1


#nvcc -std=c++14  base/PPRMain.cu -o res
#./res ../dataset/cnr-2000/origin