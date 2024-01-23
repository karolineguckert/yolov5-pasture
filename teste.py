from ultralytics import YOLO

model = YOLO("./runs/train/exp/weights/best.pt")

results = model.predict(source="./datasets/val/images")
# python train.py --img 448 --batch 16 --epochs 3 --data teste.yaml

# python detect.py --weights ./runs/train/exp/weights/best.pt --source ./datasets/val/images