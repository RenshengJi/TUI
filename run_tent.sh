
# CP_methods=("None" "TUI")
CP_methods=("TUI")
# CP_methods=("None")
CP_alphas=(0.1) # 0.2 0.1 

# for CP_alphas in ${CP_alphas[*]}; do 
#     for CP_method in ${CP_methods[*]}; do
#         python test_time.py --cfg ./cfgs/cifar10_c/tent.yaml --gpu 4 --CP_method $CP_method --CP_alpha $CP_alphas
#     done
# done


# for CP_alphas in ${CP_alphas[*]}; do 
#     for CP_method in ${CP_methods[*]}; do
#         python test_time.py --cfg ./cfgs/cifar100_c/tent.yaml --gpu 4 --CP_method $CP_method --CP_alpha $CP_alphas
#     done
# done

for CP_alphas in ${CP_alphas[*]}; do 
    for CP_method in ${CP_methods[*]}; do
        python test_time.py --cfg ./cfgs/imagenet_c/tent.yaml --gpu 5,7 --CP_method $CP_method --CP_alpha $CP_alphas
    done
done