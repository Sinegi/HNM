for aa in 1
do
    for ss in 10 20 
    do
        for ff in $(seq 0 9)
        do
            python main.py --s0 $aa --num_size $ss --random_seed $ff
        done
    done
done

# for aa in 3 4
# do
#     for ss in 10 20 
#     do
#         for ff in $(seq 0 9)
#         do
#             python ours.py --type $aa --num_size $ss --random_seed $ff
#         done
#     done
# done

# for aa in 5
# do
#     for ss in 30
#     do
#         for ff in $(seq 0 9)
#         do
#             python ours.py --type $aa --num_size $ss --random_seed $ff
#         done
#     done
# done

# for aa in 5
# do
#     for ss in 30
#     do
#         for ff in $(seq 0 9)
#         do
#             python data_generation.py --s0 $aa --num_size $ss --random_seed $ff
#         done
#     done
# done