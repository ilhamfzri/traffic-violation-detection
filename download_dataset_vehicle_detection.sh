#Create Folder to Store Vehicle Detection Dataset
mkdir Vehicle_Detection_Dataset
mkdir Vehicle_Detection_Dataset/images
mkdir Vehicle_Detection_Dataset/images/train
mkdir Vehicle_Detection_Dataset/images/val
mkdir Vehicle_Detection_Dataset/labels
mkdir Vehicle_Detection_Dataset/labels/train
mkdir Vehicle_Detection_Dataset/labels/val

#Download Kitti Dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1R1sW4Yhh2kQYDbxW_fV3N3maKYfJjh4Q' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1R1sW4Yhh2kQYDbxW_fV3N3maKYfJjh4Q" -O kitti_train_images.tar && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-1VqpNSC-uazijDK-fncwwxfzvqGCR0Y' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-1VqpNSC-uazijDK-fncwwxfzvqGCR0Y" -O kitti_val_images.tar && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-6n-hKn_dBivr1CMCJNiKTYC2zNdY4yZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-6n-hKn_dBivr1CMCJNiKTYC2zNdY4yZ" -O kitti_train_yolov5_annotations.tar && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-6vv5NjBIRYpH3xeKISji1sYwj7kTDG1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-6vv5NjBIRYpH3xeKISji1sYwj7kTDG1" -O kitti_val_yolov5_annotations.tar && rm -rf /tmp/cookies.txt

#Extract Kitti Dataset
tar -xvf "kitti_train_images.tar"
tar -xvf "kitti_val_images.tar"
tar -xvf "kitti_train_yolov5_annotations.tar"
tar -xvf "kitti_val_yolov5_annotations.tar"

#Move Kitti Dataset to Vehicle Detection Dataset Folder
mv -v kitti_train_images/* Vehicle_Detection_Dataset/images/train
mv -v kitti_val_images/* Vehicle_Detection_Dataset/images/val
mv -v kitti_train_yolov5_annotations/* Vehicle_Detection_Dataset/labels/train
mv -v kitti_val_yolov5_annotations/* Vehicle_Detection_Dataset/labels/val

#Download COCO Dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-3wORzOBNTDEh2Ih9xM0MsRzWy_TXoS3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-3wORzOBNTDEh2Ih9xM0MsRzWy_TXoS3" -O coco_train_images.tar && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-BP10dWmy3sa0VmfqtaNA1ZbP6Xx4qt1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-BP10dWmy3sa0VmfqtaNA1ZbP6Xx4qt1" -O coco_val_images.tar && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-6SGTw9NmOtxaXtqvIdkEu-VkMdmLw-1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-6SGTw9NmOtxaXtqvIdkEu-VkMdmLw-1" -O coco_train_yolov5_annotations.tar && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-I7QCacXampP2osw7LAAHPL-Cj6j_0dq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-I7QCacXampP2osw7LAAHPL-Cj6j_0dq" -O coco_val_yolov5_annotations.tar && rm -rf /tmp/cookies.txt

#Extract COCO Dataset
tar -xvf "coco_train_images.tar"
tar -xvf "coco_val_images.tar"
tar -xvf "coco_train_yolov5_annotations.tar"
tar -xvf "coco_val_yolov5_annotations.tar"

#Move COCO Dataset to Vehicle Detection Dataset Folder
mv -v coco_train_images/* Vehicle_Detection_Dataset/images/train
mv -v coco_val_images/* Vehicle_Detection_Dataset/images/val
mv -v coco_train_yolov5_annotations/* Vehicle_Detection_Dataset/labels/train
mv -v coco_val_yolov5_annotations/* Vehicle_Detection_Dataset/labels/val

#Download GTA V Dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sfmQpoQPZTJTsoaALrIrl_ok9gZFHAac' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sfmQpoQPZTJTsoaALrIrl_ok9gZFHAac" -O gtav_train_images.tar && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-0XLhS9pqNd-z7FawCB1bHA3MgNVt-iv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-0XLhS9pqNd-z7FawCB1bHA3MgNVt-iv" -O gtav_val_images.tar && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-7E_H3uo_BO6BxvITVnFyF7485P8e8WH' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-7E_H3uo_BO6BxvITVnFyF7485P8e8WH" -O gtav_train_yolov5_annotations.tar && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-7E_H3uo_BO6BxvITVnFyF7485P8e8WH' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-7XsyD0KDRyaozL3TWulHl9HUn_VbiSN" -O gtav_train_val_annotations.tar && rm -rf /tmp/cookies.txt

#Extract GTA V Dataset
tar -xvf "gtav_train_images.tar"
tar -xvf "gtav_val_images.tar"
tar -xvf "gtav_train_yolov5_annotations.tar"
tar -xvf "gtav_val_yolov5_annotations.tar"

#Move GTA V Dataset to Vehicle Detection Dataset Folder
mv -v gtav_train_images/* Vehicle_Detection_Dataset/images/train
mv -v gtav_val_images/* Vehicle_Detection_Dataset/images/val
mv -v gtav_train_yolov5_annotations/* Vehicle_Detection_Dataset/labels/train
mv -v gtav_val_yolov5_annotations/* Vehicle_Detection_Dataset/labels/val

#Remove Folder
rm -r kitti_train_images kitti_val_images coco_train_images coco_val_images gtav_train_images gtav_val_images
rm -r kitti_train_yolov5_annotations kitti_val_yolov5_annotations coco_train_yolov5_annotations coco_val_yolov5_annotations
rm -r gtav_train_yolov5_annotations gtav_val_yolov5_annotations 