import cv2 as cv
import numpy as np 

RED = (0,0,255)
p0, p1 = (10,50), (490, 50) #yatay düz bir çizgi için koordinatlar

#siyah resim
img = np.zeros((100,500,3), np.uint8)

#trackbar(sürgü) hareket ettiğinde otomatik çalışacak fonksiyon => callback 
def trackbar(x):
    #çizgi kalınlığı 0 olamaz bu yüzden en az 1 yapıyoruz.
    x = max(1, x)
    
    #yeni kalınlıkta çizgi çizmeden önce eski ekranı temizle 
    #img[:] = 0 → Resmin tüm piksellerini sıfırla (siyaha çevir). Yeniden çizmeden önce eski çizgiyi silmek için kullanılır.
    img[:] = 0 

    #yeni x kalınlığı ile çizgiyi çiz
    cv.line(img, p0, p1, RED, x)
    
    #hangi kalınlıkta olduğumuzu pencere başlığına yazdırma
    cv.setWindowTitle('pencere',f'Kalinlik: {x}')
    cv.imshow('pencere', img)

#ilk önce pencere oluştur
cv.namedWindow('pencere')

#ilkte ekranda görünecek çizgi
cv.line(img, p0, p1, RED, 2)
cv.imshow('pencere', img)

#trackbar oluşturma. Parametreler: (sürgü adı, pencere adı,başlangıç değeri,maksimum değer, teiklenecek fonksiyon)
#cv.createTrackbar('thickness', 'window', 2, 20, trackbar)
cv.createTrackbar('Kalinlik' , 'pencere', 2, 20, trackbar)
    
cv.waitKey(0)
cv.destroyAllWindows()

