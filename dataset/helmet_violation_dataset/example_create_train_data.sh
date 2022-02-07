python dataset/helmet_violation_dataset/create_train_data.py \
    --input_dir dataset/helmet_violation_dataset/author_dataset \
    --output_dir dataset/helmet_violation_dataset/final_helmet_violation_dataset \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 

python dataset/helmet_violation_dataset/create_train_data.py \
    --input_dir dataset/helmet_violation_dataset/makeml_dataset \
    --output_dir dataset/helmet_violation_dataset/final_helmet_violation_dataset \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 