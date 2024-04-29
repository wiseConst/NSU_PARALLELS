set multiplot layout 2,2
set datafile separator ','

set xlabel 'Processors'
set ylabel 'Time'
plot 'time.txt' using 1:2 with lines title 'N=512'

set xlabel 'Processors'
set ylabel 'Acceleration'
plot 'acceleration.txt' using 1:2 with lines title 'N=512'

set xlabel 'Processors'
set ylabel 'Efficiency'
plot 'efficiency.txt' using 1:2 with lines title 'N=512'
