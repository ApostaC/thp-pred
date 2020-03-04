if [ $# -eq 1 ]; then
    ./gen.sh $1
fi

gnuplot plot.gp
