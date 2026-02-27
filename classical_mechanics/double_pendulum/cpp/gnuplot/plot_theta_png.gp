set terminal pngcairo size 1200,700 font "Sans,12"
set output "theta_time.png"

set datafile separator comma
set key autotitle columnhead
set grid
set xlabel "t [s]"
set ylabel "theta [rad]"
file = "case_chaotic.csv"

plot file every ::1 using 1:2 with lines lw 2 title "theta1", \
     file every ::1 using 1:4 with lines lw 2 title "theta2"

set output
