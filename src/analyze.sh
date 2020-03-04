if [ $# -eq 1 ]; then
    ./gen.sh $1
fi

gnuplot plot.gp

#scp result.pdf yihua@netxp.cs.jhu.edu:temp/pdfs/scc-result.pdf
#scp result.pdf aposta@47.94.151.147:scc-result.pdf
