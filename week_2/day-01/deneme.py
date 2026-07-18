import cv2 as cv
from ultralytics import YOLO

# Modeli yükle
model = YOLO('yolov8m.pt') 

# Video yolu (kendi yolunu koy)
video_path = 'people_in_the_cafe.mp4'
cap = cv.VideoCapture(video_path)

# En basit takip: Hiçbir yaml dosyası yok, ekstra ayar yok
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Sadece varsayılan takipçi (BoT-SORT çalışır)
    results = model.track(frame, persist=True, classes=[0])
    
    # Çizimi de Ultralytics'in kendi fonksiyonuna yaptıralım
    annotated_frame = results[0].plot()
    
    cv.imshow("Tracking", annotated_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()