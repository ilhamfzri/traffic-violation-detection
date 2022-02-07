python yolov5_repo/train.py --img 300 \
--cfg configs/helmet_detection/helmet_violation_model_conf.yaml \
--hyp configs/helmet_detection/helmet_violation_hyper_conf.yaml \
--batch 4 \
--weight yolov5m.pt \
--epochs 20 \
--data configs/helmet_detection/helmet_violation_data_conf.yaml \
--workers 8 \
--project "/content/drive/MyDrive/Skripsi/ModelHelmetRuns" \
--save-period 1 