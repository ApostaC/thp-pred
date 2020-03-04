set terminal pdf font 'Helvetica 20'
set output 'result.pdf'
set grid

set xlabel 'epoch'
plot 'running.csv' u 1:2 w l title 'train loss', 'running.csv' u 1:3 w l title 'test loss'

set xlabel 'time (sec)'
set ylabel 'relative thp'
plot 'test.txt' u 1:2 w l title 'real', 'test.txt' u 1:3 w l title 'pred'

set xlabel 'time (sec)'
set ylabel 'relative thp'
plot 'plot1.txt' u 1:2 w l title 'real', 'plot1.txt' u 1:3 w l title 'pred' lw 2

set xr [0:]
set xlabel 'time (sec)'
set ylabel 'relative thp'
plot 'plot1.txt' u 1:2 w l title 'real', 'plot1.txt' u ($1-5):3 w l title 'pred-shifted' lw 2
