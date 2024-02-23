from ultralytics import YOLO

model = YOLO("./runs/train/exp3/weights/best.pt")

results = model.predict("./datasets/val/images/P3720423.jpg")
# python train.py --img 448 --batch 16 --epochs 3 --data teste.yaml
# python train.py --img 4608 --batch 16 --epochs 50 --data teste.yaml
# python detect.py --weights ./runs/train/exp/weights/best.pt --img 4608 --conf 0.4 --source ./datasets/val/images/tamanho1
#python detect.py --weights weights/last_yolov5s_custom.pt --img 416 --conf 0.4 --source ../test_infer