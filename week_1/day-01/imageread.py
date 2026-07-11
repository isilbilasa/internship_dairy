import cv2 as cv

#1.reading the images using imread() function
#cv2.imread() fonksiyonu resmi bir matris(numpy array) olarak yükler. resmi diskten oku, "img" değişkenine koy
#imread = "image read" (resim oku). Diskteki dosyayı belleğe alır.
image = cv.imread('deneme.jpg') 

#extracting the height and weight of an image 
h,w = image.shape[:2]
#displaying the height and weight 
print("Height={}, Width={}".format(h,w))

#kontrol: resim doğru şekilde yüklendi mi
if image is None:
    print("Hata: Resim bulunamadı! Lütfen dosya yolunu kontrol edin.")
else:
    #2.resmi yeni bir pencerede gösterme
    #ilk parametre pencerenin adı,ikincisi ise yüklü olan resim değişkenidir.
    # imshow = "image show" (resmi göster). 'window' pencerenin adıdır.
    cv.imshow("Ilk OpenCv Denemesi",image)

    #3.pencerenin ekranda kalmasını sağlama 
    #cv2.waitKey(0)klavyeden herhangi bir tuşa basılana kadar pencereyi açık tutar 
    #parametre olarak milisaniye alır; 0 verilirse sonsuza kadar bekler.
    # 0 = sonsuza kadar bekle. bu çağrılmazsa pencere görünmez.bu parametre, bir tuşa basılması için kaç milisaniye bekleneceğidir.
    cv.waitKey(0)

    #4.resmi yeni bir isimle kaydetme 
    #"image write" (resmi yaz/kaydet). 
    # Uzantıyı (.jpg, .png, .tiff) siz seçersiniz; OpenCV otomatik olarak o formata çevirir.
    cv.imwrite("kopya_resim2.png",image)
    print("resim başarıyla kaydedildi")

    #işlem bittiğinde açılan tüm OpenCV pencereleri kapatır
    cv.destroyAllWindows()

    