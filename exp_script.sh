
count=1
for e_l in 2 3 4 5 6
do
    for lr in 0.00008 0.00007 0.00006 0.00005 0.00004 0.00003 0.00002 0.00001
    do 
        CUDA_VISIBLE_DEVICES=2 python main.py \
        --config-file './configs/PatchTST.yaml' \
        --opts model.e_layers $e_l optimizer.learning_rate $lr logger.run_name b$count
        count=$((count+1))
    done
done