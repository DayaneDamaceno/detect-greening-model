import os
import json
from urllib.parse import urlparse
from ultralytics import YOLO

def init():
    global model
    # Carrega o modelo do caminho registrado no Azure ML
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "best.pt")
    model = YOLO(model_path)

def validate_input(image_urls):
    """
    Valida se a entrada é uma lista de strings representando URLs.
    """
    if not isinstance(image_urls, list) or len(image_urls) == 0:
        return False, json.dumps({"error": "A entrada deve ser uma lista de URLs de imagens."}, indent=4)
    for idx, url in enumerate(image_urls):
        if not isinstance(url, str):
            return False, json.dumps({"error": f"Cada item na lista deve ser uma string representando um URL. Erro no índice {idx}."}, indent=4)
        if not (url.startswith("http://") or url.startswith("https://")):
            return False, json.dumps({"error": f"O URL no índice {idx} não parece válido: {url}"}, indent=4)
    return True, None

def get_image_name_from_url(image_url):
    """
    Extrai o nome do arquivo da URL da imagem.
    """
    return os.path.basename(urlparse(image_url).path)

def run(image_urls):
    """
    Espera-se que `image_urls` seja uma lista de strings, onde cada string é um URL de imagem.
    """
    is_valid, error_response = validate_input(image_urls)
    if not is_valid:
        return error_response

    try:
        # Chama o método predict passando a lista de URLs diretamente
        predictions = model.predict(image_urls, conf=0.5, save=False, save_txt=False)
        
        final_results = []
        for idx, pred in enumerate(predictions):
            image_name = get_image_name_from_url(image_urls[idx])  # Extrai o nome da imagem a partir da URL
            detections = []
            if hasattr(pred, "boxes") and pred.boxes is not None:
                for box in pred.boxes:
                    detections.append({
                        "label": pred.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist()
                    })
            final_results.append({
                "id": image_name,  # A chave 'id' agora contém o nome da imagem
                "detections": detections
            })
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=4)

    return json.dumps(final_results, indent=4)
