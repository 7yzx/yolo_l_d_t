from ultralytics import YOLO
import argparse
import os


if __name__ == "__main__":
    paraser = argparse.ArgumentParser()
    paraser.add_argument('--weights_path', "-W", type=str, required=True, default="", help='model.pt path(s), i.e. yolov5s.pt')
    paraser.add_argument('--output_path', type=str, default="", help='output path , default=weights_path')
    
    args = paraser.parse_args()
    
    model_path = os.path.normpath(args.weights_path)
    output_path = args.output_path
    model = YOLO(model_path)  # load a custom trained model

    # Export the model
    model.export(format="onnx")