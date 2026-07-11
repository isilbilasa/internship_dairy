import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys

# güvenlik kontrolü 
dosya_yolu = 'images-3.png'
img = cv.imread(dosya_yolu)

if img is None:
    print(f"HATA: '{dosya_yolu}' dosyası bulunamadı! Lütfen yolu kontrol et.")
    sys.exit()


# BÖLÜM 1: RENKLİ HİSTOGRAM (BGR KANALLARI)

# Görüntüdeki Mavi, Yeşil ve Kırmızı yoğunluğunu ayrı ayrı grafiğe döküyoruz.
renkler = ('b', 'g', 'r') # OpenCV BGR formatında okuduğu için sıra böyledir

# Matplotlib'de yeni bir grafik penceresi açıyoruz
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1) # 1 satır, 2 sütunlu grafiğin 1.si
plt.title("Renkli Histogram (BGR)")

for i, renk in enumerate(renkler):
    # calcHist parametreleri: [görsel], [kanal_no], maske, [kutu_sayisi], [deger_araligi]
    hist_renk = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist_renk, color=renk)
    plt.xlim([0, 256])


# BÖLÜM 2: GRİ HİSTOGRAM VE KONTRAST ARTIRMA (EQUALIZATION)

# Histogram eşitleme işlemi (görüntüyü iyileştirme) sadece tek kanallı (gri) resimlerde yapılır.
gri_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Histogram Eşitleme: Işığı görüntünün her yerine dengeli dağıtır (Kontrastı artırır)
esitlenmis_img = cv.equalizeHist(gri_img)

# Orijinal gri ve eşitlenmiş gri resimlerin histogramlarını hesaplayalım
hist_orijinal_gri = cv.calcHist([gri_img], [0], None, [256], [0, 256])
hist_esitlenmis_gri = cv.calcHist([esitlenmis_img], [0], None, [256], [0, 256])

# İkinci grafiği çizelim
plt.subplot(1, 2, 2)
plt.title("Gri Histogram Eşitleme")
plt.plot(hist_orijinal_gri, color='black', label='Orijinal')
plt.plot(hist_esitlenmis_gri, color='green', label='Eşitlenmiş', linestyle='--')
plt.legend()
plt.xlim([0, 256])


# BÖLÜM 3: EKRANDA GÖSTERİM

# Resimleri yan yana birleştirip tek pencerede göstermek için yatay yığma (hstack) yapıyoruz
# Not: Eşitleme işleminin görüntüde yarattığı farka dikkat et!
yan_yana_resimler = np.hstack((gri_img, esitlenmis_img))
cv.imshow('Orijinal Gri VS Kontrasti Artirilmis Gri', yan_yana_resimler)

print("Grafiklerin bulunduğu Matplotlib penceresini kapattığında program sonlanacaktır.")

# Grafikleri ekrana bas (Bu komut çalışınca grafik penceresi kapanana kadar kod bekler)
plt.show()

# Pencereleri temizle
cv.waitKey(0)
cv.destroyAllWindows()