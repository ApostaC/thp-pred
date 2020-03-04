#########################################################################
# File Name: gen.sh
# Author: Jing
# mail: jing.wang@pku.edu.cn
# Created Time: 2019年05月22日 星期三 10时57分35秒
#########################################################################
#!/bin/bash
if [ $# -ne 1 ]; then
	echo "Usage $0 <starttime>"
	exit
fi
echo "Start = $1"
awk -v st=$1 '{if ($1 >= st && $1 <= st + 200) print $1 - st, $2, $3}' test.txt > plot1.txt
