import os
import json
from ultralytics import YOLO

def init():
    global model

    # Carrega o modelo do caminho registrado no Azure ML
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "best.pt")
    model = YOLO(model_path)

def run(images):
    # Valida a entrada para evitar erros inesperados
    if not isinstance(images, list) or len(images) == 0:
        return json.dumps({"error": "A entrada deve ser uma lista de caminhos de imagens."}, indent=4)

    results = []
    
    try:
        # Processa todas as imagens do lote de uma vez
        predictions = model.predict(images, conf=0.5, save=False, save_txt=False)  

        for image_path, pred in zip(images, predictions):
            detections = []

            if hasattr(pred, "boxes") and pred.boxes is not None:
                for box in pred.boxes:
                    detections.append({
                        "label": pred.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist()
                    })

            # Garante que todas as imagens têm um resultado, mesmo sem detecções
            results.append({
                "image": image_path,
                "detections": detections
            })
    
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=4)

    # Retorna a lista completa em formato JSON
    return json.dumps(results, indent=4)
