import cv2 as cv
import numpy as np

cap = cv.VideoCapture("red_ball.mp4")

#kırmızı için HSV alt ve üst değerleri. Kırmızı HSV'de hem 0 hem de 180 derecenin etrafında olduğu için iki aralık oluyor. 
lower_red = np.array([0, 50, 50]) 
upper_red = np.array([15, 255, 255])

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))

if fps == 0:
    fps = 30

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('top_takibi_sonuc.mp4', fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # görüntüyü noisedan engellemek için hafif bulanıklaştırdık
    blurred = cv.GaussianBlur(frame, (11, 11), 0)
    
    # bgr => hsv geçiş 
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_red, upper_red)

    # erozyon ve genişletme => içteki boşluklar dolması için
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=3) 

    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(c)

        M = cv.moments(c)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # eğer yarıçap belirli bir boyuttan büyükse 
            if radius > 10:
                # topun etrafına çember
                cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv.circle(frame, center, 5, (0, 0, 255), -1)
                
     # cv.imshow("Maske (Siyah Beyaz)", mask)
                
    cv.imshow("Top takibi", frame)
    out.write(frame)

    if cv.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
out.release() 
cv.destroyAllWindows()