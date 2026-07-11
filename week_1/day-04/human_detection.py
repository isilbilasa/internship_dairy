import cv2 as cv
from ultralytics import YOLO
import datetime

model = YOLO('yolov8m.pt')

video_path = 'people_in_the_cafe.mp4'
cap = cv.VideoCapture(video_path)

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))

fourcc = cv.VideoWriter_fourcc(*'avc1')

time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"cafe_detection_{time_stamp}.mp4"

out =cv.VideoWriter(file_name,fourcc,fps,(frame_width,frame_height))

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break    
    #modeli çalıştır => classes=[0] sadece insanları filtreler. ve conf = 0.5 parametresi %50'den düşük doğruluk oranlarını eler 
    results = model(frame, classes= [0], conf = 0.3)

    #modelin bulduğu kutuları çercevenin üstüne çiz.
    annotative_frame = results[0].plot()
    
    out.write(annotative_frame)        #işlenmiş olan kareyi oluşan video içine yazıyorum.

    #çıktıyı ekranda göster.
    cv.imshow('Kafede insan tespiti', annotative_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv.destroyAllWindows()