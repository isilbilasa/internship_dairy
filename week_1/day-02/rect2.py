import cv2 as cv
import numpy as np

#renkleri verelim 
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)

#dikdörtgenin karşılıklı bölgeleri 
#(x,y) koordinatları
p0, p1 = (50,50), (250,150)     #ilk dikdörtgenin sol üst ve sağ alt köşesi
p2, p3 = (300,200),(450,300)    #ikinci dikdörtgenin sol üst ve sağ alt köşesi 

img = np.zeros((400,600,3), np.uint8)

#sadece çercevesi oan dikdörtgen çizimi (kalınlık: 2)
cv.rectangle(img,p0,p1,BLUE,2)
#içi tamamen dolu didörtgen çizimi (Klaınlık: cv.FILLED veya -1)
cv.rectangle(img,p2,p3,GREEN,cv.FILLED)

#ana program
cv.imshow('Pencere',img)

print("mavi:sadece çerceve ola dikdörtgen")
print("yeşil:içi dolu(filled) dikdörtgen")

cv.waitKey(0)
cv.destroyAllWindows()