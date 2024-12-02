

exec_read=../bin/convert2binary

input_file_path=/home/lihuaibei/code/stanford-pr
output_directory_path=../dataset/stanford
#$exec_read $input_file_path $output_directory_path


#exec_sssp=../bin/purn-sssp
#$exec_sssp $output_directory_path
#
#ssspbase=../bin/sssp-base
#
#$ssspbase $output_directory_path 1
#
#sssppurn=../bin/sssp-purn
#
#$sssppurn $output_directory_path 1



#exec_sssp=../bin/purn-ppr
#
#$exec_sssp $output_directory_path
#
#pprbase=../bin/ppr-base
#
#$pprbase $output_directory_path 5
#
#pprpurn=../bin/ppr-purn
#
#$pprpurn $output_directory_path 5




#ssspbaseM=../bin/sssp-base-M
#
#$ssspbaseM $output_directory_path 5 40

sssppurnM=../bin/sssp-purn-M

$sssppurnM $output_directory_path 5 40