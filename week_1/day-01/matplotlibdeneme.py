import cv2
import matplotlib.pyplot as plt

#burda görüntüyü opencv ile okuduk(bgr formatında gelicek)
foto = cv2.imread("kopya_resim.png")

#img[baslangic_Y : bitis_Y, baslangıc_X : bitis_X]
kirpilmis_foto = foto[50:100, 100:300]
#resize cv2.resize(görüntü,(yeni_genislik, yeni_yükseklik))
resize_image = cv2.resize(foto, (10,100))

#matplotlib için renkleri düzelticem (bgr den rgb ye çevirme işlemi) 
# cvtColor = "convert color"(renk uzayına çevir)
#foto_rgb = cv2.cvtColor(foto,cv2.COLOR_BGR2RGB)

#cv.COLOR_BGR2GRAY = "BGR'den GRAY'e" anlamına gelen bir sabittir (hazır tanımlı bir ayar).
gray = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY) #BGR -> Gri dönüşümü

#görüntüyü matplotlibe vericem 
plt.imshow(kirpilmis_foto)
plt.imshow(resize_image)
plt.imshow(foto)

cv2.imwrite("resim_gray.png",gray)
print(foto.shape)
print(foto.shape[0]) #yükseklik(satır sayısı=piksel olarak boy)
print(foto.shape[1]) #genişlik(sütun sayısı=piksel olarak en)
print(foto.shape[2])#kanal sayısı(renkli resimde 3: B,G,R)

#x ve y eksenlerini saklıyor => hide all components of the x-and y-axis
plt.axis('off')
plt.show()

