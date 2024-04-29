set multiplot layout 2,2
set datafile separator ','

set xlabel 'Threads'
set ylabel 'Acceleration'
plot 'var_2_accel.txt' using 1:2 with lines title 'Var_2'

set xlabel 'Threads'
set ylabel 'Execution Time(seconds)'
plot 'var_2_raw.txt' using 1:2 with lines title 'Var_2'

set xlabel 'Threads'
set ylabel 'Efficiency'
plot 'var_2_efficiency.txt' using 1:2 with lines title 'Var_2'
