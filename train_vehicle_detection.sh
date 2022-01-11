python yolov5/train.py --img 640 \
--cfg configs/vehicle_detection/vehicle_detection_model_conf.yaml \
--hyp configs/vehicle_detection/vehicle_detection_hyper_conf.yaml \
--batch -1 \
--weight yolov5m.pt \
--epochs 100 \
--data configs/vehicle_detection/vehicle_detection_data_conf.yaml \
--workers 8 \
--project "/content/drive/MyDrive/Skripsi/Model Vehicle Detection" \
--save-period 1 
