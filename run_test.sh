# CP_methods=("None" "THR" "NexCP" "QTC" "TUI")
# CP_alphas=(0.3 0.2 0.1)

CP_methods=("TUI")
CP_alphas=(0.2)

for CP_alphas in ${CP_alphas[*]}; do
    for CP_method in ${CP_methods[*]}; do
        python test_time_plot.py --cfg ./cfgs/cifar100_c/CPCTTA.yaml --gpu 0 --CP_method $CP_method --CP_alpha $CP_alphas
    done
done
