from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='gates_dataset.yaml',
        epochs=100,
        imgsz=640,
        device='cpu'
    )

if __name__ == '__main__':
    main()
    