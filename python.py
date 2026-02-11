import os 
from imageai.Detection import ObjectDetection  

PROJECT_NAME = "SafetyAI"
MODEL_FILE = "yolov3.pt"
INPUT_IMAGE = "road.jpg"
OUTPUT_IMAGE = "road_detected.jpg"

def print_project_header():
    print(f"🚀 Запуск системы {PROJECT_NAME}...")

def detect_objects_on_road(input_path, output_path):
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(MODEL_FILE)
    detector.loadModel()
    detector.detectObjectsFromImage(
        input_image=input_path, 
        output_image_path=output_path, 
        minimum_percentage_probability=30
    )

print_project_header()
detect_objects_on_road(INPUT_IMAGE, OUTPUT_IMAGE)