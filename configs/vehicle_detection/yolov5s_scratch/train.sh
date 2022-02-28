python yolov5/train.py --img 640 \
--cfg configs/vehicle_detection/yolov5s_scratch/vehicle_detection_model_conf.yaml \
--hyp configs/vehicle_detection/yolov5s_scratch/vehicle_detection_hyper_conf.yaml \
--batch 64 \
--weight '' \
--epochs 100 \
--data configs/vehicle_detection/yolov5s_scratch/vehicle_detection_data_conf.yaml \
--workers 8 \
--project "/content/drive/MyDrive/Skripsi/Model Vehicle Detection" \
--save-period 5