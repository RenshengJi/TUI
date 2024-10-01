CP_methods=("TUI") #  "None" 
CP_alphas=(0.2) #  0.2 0.1

# for CP_alphas in ${CP_alphas[*]}; do 
#     for CP_method in ${CP_methods[*]}; do
#         python test_time.py --cfg ./cfgs/cifar10_c/CCoTTA.yaml --gpu 2 --CP_method $CP_method --CP_alpha $CP_alphas --cav_alpha 2.0 --cav_beta 0.05
#     done
# done

# for CP_alphas in ${CP_alphas[*]}; do 
#     for CP_method in ${CP_methods[*]}; do
#         python test_time.py --cfg ./cfgs/cifar100_c/CCoTTA.yaml --gpu 6 --CP_method $CP_method --CP_alpha $CP_alphas --cav_alpha 1.0 --cav_beta 0.01
#     done
# done

for CP_alphas in ${CP_alphas[*]}; do 
    for CP_method in ${CP_methods[*]}; do
        python test_time.py --cfg ./cfgs/imagenet_c/CCoTTA.yaml --gpu 1,3 --CP_method $CP_method --CP_alpha $CP_alphas --cav_alpha 1.0 --cav_beta 0.05
    done
done