import os
import json
import numpy as np
import cv2
# Se necessário, importe base64 para tratar strings codificadas:
# import base64
from ultralytics import YOLO

def init():
    global model
    # Carrega o modelo do caminho registrado no Azure ML
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "best.pt")
    model = YOLO(model_path)

def validate_input(image_binaries):
    """
    Valida a entrada fornecida.
    """
    if not isinstance(image_binaries, list) or len(image_binaries) == 0:
        return False, json.dumps({"error": "A entrada deve ser uma lista de dados binários de imagens."}, indent=4)
    return True, None

def process_images(image_binaries):
    """
    Converte cada item da lista para um array NumPy (imagem no formato OpenCV).
    """
    results = []
    images_for_prediction = []
    
    for idx, image_bin in enumerate(image_binaries):
        try:
            # Caso você receba imagens em base64, descomente as linhas abaixo:
            # if isinstance(image_bin, str):
            #     image_bin = base64.b64decode(image_bin)
            
            nparr = np.frombuffer(image_bin, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_np is None:
                raise ValueError(f"Não foi possível decodificar a imagem no índice {idx}.")
            images_for_prediction.append(img_np)
        except Exception as e:
            results.append({
                "index": idx,
                "error": f"Erro ao processar a imagem: {str(e)}"
            })
            images_for_prediction.append(None)
    
    return images_for_prediction, results

def filter_valid_images(images_for_prediction):
    """
    Filtra as imagens válidas para previsão e guarda os índices correspondentes.
    """
    valid_indices = [i for i, img in enumerate(images_for_prediction) if img is not None]
    valid_images = [img for img in images_for_prediction if img is not None]
    return valid_indices, valid_images

def predict_images(valid_images):
    """
    Executa a previsão em lote com as imagens válidas.
    """
    return model.predict(valid_images, conf=0.5, save=False, save_txt=False)

def map_predictions_to_results(valid_indices, predictions):
    """
    Mapeia os resultados às imagens válidas pelo índice original.
    """
    valid_results = {}
    for valid_idx, pred in zip(valid_indices, predictions):
        detections = []
        if hasattr(pred, "boxes") and pred.boxes is not None:
            for box in pred.boxes:
                detections.append({
                    "label": pred.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()
                })
        valid_results[valid_idx] = detections
    return valid_results

def consolidate_results(image_binaries, valid_results):
    """
    Consolida os resultados para todas as imagens, mantendo o índice original.
    """
    final_results = []
    for idx in range(len(image_binaries)):
        if idx in valid_results:
            final_results.append({
                "index": idx,
                "detections": valid_results[idx]
            })
        else:
            final_results.append({
                "index": idx,
                "error": "Imagem inválida ou não processada."
            })
    return final_results

def run(image_binaries):
    """
    Espera-se que `image_binaries` seja uma lista onde cada elemento é o conteúdo binário de uma imagem.
    """
    is_valid, error_response = validate_input(image_binaries)
    if not is_valid:
        return error_response

    images_for_prediction, results = process_images(image_binaries)
    valid_indices, valid_images = filter_valid_images(images_for_prediction)
    
    if not valid_images:
        return json.dumps({"error": "Nenhuma imagem válida foi fornecida."}, indent=4)
    
    try:
        predictions = predict_images(valid_images)
        valid_results = map_predictions_to_results(valid_indices, predictions)
        final_results = consolidate_results(image_binaries, valid_results)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=4)

    return json.dumps(final_results, indent=4)
