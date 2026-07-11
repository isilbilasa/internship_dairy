import cv2 as cv
import numpy as np

img = cv.imread('images-3.png')
yukseklik, genislik = img.shape[:2]

# Translation
# Görseli X ekseninde (sağa) 100 piksel, Y ekseninde (aşağı) 50 piksel kaydıracağız.
# Bunun için 2x3 boyutunda bir Numpy matrisi (T) oluşturmalıyız.
# Matrisin yapısı: [[1, 0, x_kayma], [0, 1, y_kayma]]
T_matrisi = np.float32([[1, 0, 100], 
                        [0, 1, 50]])

# warpAffine parametreleri: (görsel, dönüşüm_matrisi, (yeni_genislik, yeni_yukseklik))
kaydirilmis_img = cv.warpAffine(img, T_matrisi, (genislik, yukseklik))


# Rotation
# Görseli tam merkezinden 45 derece döndürelim ve boyutunu %50 küçültelim.
merkez_noktasi = (genislik // 2, yukseklik // 2)
aci = 45
olcek = 0.5 # 1.0 yaparsan orijinal boyutunda döner

# OpenCV bizim için döndürme matrisini (R) otomatik hesaplar
R_matrisi = cv.getRotationMatrix2D(merkez_noktasi, aci, olcek)

# Hesaplanan matrisi yine warpAffine'a veriyoruz
dondurulmus_img = cv.warpAffine(img, R_matrisi, (genislik, yukseklik))


# Sonuçları ekranda göster
cv.imshow('Orijinal', img)
cv.imshow('Kaydirilmis (100 saga, 50 asagi)', kaydirilmis_img)
cv.imshow('Dondurulmus (45 derece)', dondurulmus_img)

cv.waitKey(0)
cv.destroyAllWindows()