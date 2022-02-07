python dataset/helmet_violation_dataset/preprocessing.py \
    --input_dir dataset/helmet_violation_dataset/author_dataset_raw \
    --output_dir dataset/helmet_violation_dataset/author_dataset \
    --vehicle_detection_model models/vehicle_detection/best.pt \
    --conf_thres 0.25 \
    --iou_thres 0.45

python dataset/helmet_violation_dataset/preprocessing.py \
    --input_dir dataset/helmet_violation_dataset/makeml_dataset_raw \
    --output_dir dataset/helmet_violation_dataset/makeml_dataset \
    --vehicle_detection_model models/vehicle_detection/best.pt \
    --conf_thres 0.25 \
    --iou_thres 0.45