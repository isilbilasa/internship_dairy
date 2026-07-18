import cv2 as cv

#haar cascade yüz modelini yükledik
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(1)

print("kamera açılıyor...çıkmak için 'q' tuşun basınız.")

while True:
    #kameradan anlık frame'i(kareyi) oku
    ret,frame = cap.read()
    if not ret:
        print("kameradan görüntü alınamadı")
        break

    #haar cascade gri tonlamalı(siyah-beyaz) resimlerde çalışır.
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    #yüzleri tespit et(şelale algoritması)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,     #sliding window her adımda %10 büyür.
        minNeighbors=5,     #bir kutunun yüz kabul edilmesi için gereken doğrulama sayısı,
        minSize=(30,30)     #aranacak en küçük yüz boyutu(çok uzaktaki küçük yüzleri eler.)        
    )

    #bulunan her yüz için (x, y,genişlik, yükseklik) koordinatlarını al
    for(x, y, w, h) in faces:
        #orijinal renkli görüntü üzerine yeşil dikdörtgen çiz.
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow('Yüz Tespiti', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()