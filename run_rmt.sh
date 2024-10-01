CP_methods=("TUI") #"None" 
CP_alphas=(0.3 0.2 0.1)

# for CP_alphas in ${CP_alphas[*]}; do 
#     for CP_method in ${CP_methods[*]}; do
#         python test_time.py --cfg ./cfgs/cifar10_c/rmt.yaml --gpu 7 --CP_method $CP_method --CP_alpha $CP_alphas
#     done
# done

for CP_alphas in ${CP_alphas[*]}; do 
    for CP_method in ${CP_methods[*]}; do
        python test_time.py --cfg ./cfgs/cifar100_c/rmt.yaml --gpu 2 --CP_method $CP_method --CP_alpha $CP_alphas
    done
done

# for CP_alphas in ${CP_alphas[*]}; do 
#     for CP_method in ${CP_methods[*]}; do
#         python test_time.py --cfg ./cfgs/imagenet_c/rmt.yaml --gpu 4,5 --CP_method $CP_method --CP_alpha $CP_alphas
#     done
# done