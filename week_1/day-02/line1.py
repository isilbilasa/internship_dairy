import cv2 as cv 
import numpy as np

#çizgi çizme
#cv.line(image, p0, p1, color, thickness)
#sıfırdan boş bir resmi numpy ile üretiyoruz

#renkleri tanımla(BGR şeklinde)
RED     = (0, 0, 255)        
YELLOW  = (0, 255, 255)     

#çizgilerin başlangıç orta ve bitiş noktalarını(x,y) belirle 
p0, p1 = (10,20) ,(250, 20)
p2, p3 = (10,70) ,(400, 70)

#Renkli resim yani 3 kanallı üzerine çizim.
#100 piksel yükseklik, 500 piksel genişliğinde siyah resim oluşturduk 
img = np.zeros((100, 500, 3), np.uint8)

#çizgileri çizelim 
cv.line(img, p0, p1, RED, 2)  #p0 dan p1 e 2 piksel kalınlığında kırmızı çizgi 
cv.line(img, p2, p3, YELLOW, 5)

#Gri resim yani tek kanallı üzerine çizim
#aynı boyutlarda ama gri tonlamalı siyah resim oluşturduk 
gray_img = np.zeros((100,500), np.uint8)

#gri resme renk veremeyiz, bunun yerine 0(siyah) ile 255(beyaz) arası parlaklık değeri veririz
cv.line(gray_img, p0, p1, 100, 2)   #p0 dan p1 e orta gri (100) çizgi
cv.line(gray_img, p0, p1, 255, 5)   #p1 den p2 ye tam beyaz (255) çizgi

#sonuçları ekranda gösterme kısmı show ile 
cv.imshow('Renkli cizgiler', img)
cv.imshow('Gri cizgiler', gray_img)

print("pencereler açıldı")
print("kapatmak için pencerelerden biri seçiliyken herhangi bir tuşa bas")

cv.waitKey(0)
cv.destroyAllWindows()