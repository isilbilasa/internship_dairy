import cv2 as cv 

img = cv.imread('images-3.png')

yukseklik, genislik = img.shape[:2]

#resize(yeniden boyutlandırma)
#görseli orijinal boyutunun tam yarısına düşürelim.(genişlik,yükseklik) sırasıyla verilir.
kucuk_img = cv.resize(img, (genislik // 2, yukseklik // 2), interpolation=cv.INTER_AREA)

#Flip-Aynalama
#parametreler: 1 = Yatay(Y ekseninde), 0 = Dikey (X  ekseninde), -1 = Her ikisi 
aynalanmis_img = cv.flip(img,1)

#Crop(Kırpma)
#OpenCV de özel bir kırpma fonk yok.Numpy'ın slicing özelliği kullanılır. 
#sıralama her zaman [y_baslangic : y_bitis, x_baslangic : x_bitis]şeklinde 
#sol üstten (0,0) başlayıp 300x300 piksellik bir alan keselim 

kirpilmis_img = img[0:300, 0:300]

cv.imshow('1-orijinal',img)
cv.imshow('2-küçültülmüş',kucuk_img)
cv.imshow('3-yatay aynalanmış',aynalanmis_img)
cv.imshow('4-Kirpilmis',kirpilmis_img)

cv.waitKey(0)
cv.destroyAllWindows()