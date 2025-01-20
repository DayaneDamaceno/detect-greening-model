# pip install ultralytics opencv-python matplotlib


from ultralytics import YOLO
import os

# 185 de imagens para validar
# 100 imagens saudaveis
# 100 imagens com greening

arquivo_config = 'configs_modelo.yaml'

model = YOLO('yolov8s.yaml')

resultados = model.train(
    data=arquivo_config, 
    epochs=10, 
    imgsz=640, 
    name='yolov8s_modelo')

dir_resultado = os.path.join('runs', 'detect', 'yolov8s_modelo')
print(f"Resultados salvos em: {dir_resultado}")


# model.export(format='onnx')

