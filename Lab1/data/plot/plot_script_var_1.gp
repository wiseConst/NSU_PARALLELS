set multiplot layout 2,2
set datafile separator ','

set xlabel 'Processors'
set ylabel 'Time'
plot 'cluster_1.txt' using 1:2 with lines title 'Var_1'

set xlabel 'Processors'
set ylabel 'Acceleration'
plot 'cluster_2.txt' using 1:2 with lines title 'Var_1'

set xlabel 'Processors'
set ylabel 'Efficiency'
plot 'cluster_3.txt' using 1:2 with lines title 'Var_1'
