# train deepar model on original data. Date: 2024-02-07
CUDA_VISIBLE_DEVICES=0 python train_val_test.py --cfg_file ./configs/deepar_original.yaml

# # train deepar model on correlation selected data. Date: 2024-02-02
# CUDA_VISIBLE_DEVICES=0 python train_val_test.py --cfg_file ./configs/deepar_correlation.yaml