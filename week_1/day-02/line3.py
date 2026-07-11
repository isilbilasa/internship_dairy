import cv2 as cv 
import numpy as np

#renk paletlerini belirle
RED     = (0, 0, 255)
GREEN   = (0, 255, 0)
BLUE    = (255, 0, 0)
CYAN    = (255, 255, 0)
MAGENTA = (255, 0, 255)
YELLOW  = (0, 255, 255)
WHITE   = (255, 255, 255)

#tüm renkleri tuple(liste) içine koyuyoruz. 0:RED
colors = (RED,GREEN,BLUE,CYAN,MAGENTA,YELLOW,WHITE)

p0, p1 = (10,50), (490,50)
img = np.zeros((100,500,3), np.uint8)

# trackbar fonk
def trackbar(x):
    #x sürgüden gelen değer (0 ile 6 arasında)
    #colors listesinden o sıradaki rengi seçiyoruz.
    secilen_renk = colors[x]

    img[:] = 0 #ekran temizler

    cv.line(img, p0, p1, secilen_renk, 10)

    cv.setWindowTitle('pencere', f'Renk Indeksi:{x}')
    cv.imshow('pencere', img)

#pencere oluşturma
cv.namedWindow('pencere')

#ilk açılış görüntüsü(o.indeks olan kırmızı ile başlar)
cv.line(img,p0,p1,RED,10)
cv.imshow('pencere', img)

#renk trackbarı ekleme => toplam 7 renk var maks değer 6 
cv.createTrackbar('Renk Sec', 'pencere', 0, 6, trackbar)
cv.waitKey(0)
cv.destroyAllWindows()