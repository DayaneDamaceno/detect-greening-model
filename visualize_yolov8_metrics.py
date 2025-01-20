def visualize_yolov8_metrics():
    dir_resultado_val = 'runs/detect/yolov8s_modelo_eval'

    imgs = ['F1_curve.png', 'PR_curve.png', 'P_curve.png', 'R_curve.png']
    plt.figure(figsize=(18,14))
    for i, img in enumerate(imgs):
  #print(i, img)
      grafico = cv2.imread(os.path.join(dir_resultado_val, img))
  #print(grafico)
      grafico = cv2.cvtColor(grafico, cv2.COLOR_BGR2RGB)
      plt.subplot(2, 2, i + 1)
      plt.title(imgs[i])
      plt.imshow(grafico)
      plt.axis('off')
    plt.show()