# STAJ 1. HAFTA RAPORU

**Hafta 1**

**Konu:** Computer Vision (OpenCV) ile Nesne Tespiti ve Takibi

**06 – 10 Temmuz 2026**

---

## İçindekiler

1. Kısa Özet
2. Yapılan İşler
   - Gün 1 — 06.07.2026: OpenCV Temelleri, Piksel Matematiği ve Renk Uzayları
   - Gün 2 — 07.07.2026: Çizim, İnteraktif Arayüzler ve Nesne Tabanlı Programlama
   - Gün 3 — 08.07.2026: Renk Tabanlı Drone Tespiti ve Takibi (Uygulama Projesi)
   - Gün 4 — 09.07.2026: YOLO ile Derin Öğrenme Tabanlı İnsan Tespiti ve Takibi
   - Gün 5 — 10.07.2026: Nesne Tespit Algoritmaları Teorisi, YOLO Mimarisi ve SAHI Entegrasyonu
3. Öğrenilen Kavramlar ve Teorik Analizler
   - 3.1. Nesne Tespit Algoritmalarına Genel Bakış
   - 3.2. YOLO Nedir ve Nerelerde Kullanılır?
   - 3.3. YOLO Nasıl Çalışır ve Mimarisi Nasıldır?
   - 3.4. SAHI (Slicing Aided Hyper Inference) ile "Model Körlüğünü" Aşmak
4. Karşılaşılan Zorluklar / Öğrenilenler
5. Sorular / İhtiyaç Duyduğum Destek
6. Gelecek Hafta Planı

---

## 1. Kısa Özet

Bu hafta OpenCV ile bilgisayarlı görünün temel yapı taşlarını (piksel matrisleri, koordinat sistemi, renk uzayı dönüşümleri, geometrik dönüşümler, çizim/arayüz fonksiyonları ve histogram analizi) öğrenilmiş ve bu kavramlar üç uygulamalı projede test edilmiştir: renk tabanlı drone tespiti/takibi (HSV maskeleme, morfolojik işlemler ve kontur analizi ile klasik CV), YOLO ile kafe ortamında yakın çekim insan tespiti (ID takibi ve NMS optimizasyonu) ve Üsküdar'da uzak/geniş açılı bir kamerada insan tespiti (çözünürlük parametresiyle küçük nesne tespitini iyileştirme). Süreç boyunca Google Colab ortamı ve bu ortamın VS Code'a bağlanarak çalışılması öğrenilmiştir. En çok zorlanılan nokta, değişken ışık koşullarında (gölge, lens parlaması, şehir arka planı) renk tabanlı maskelemenin kararsız kalmış olması; bu durum geometrik filtreleme (alan ve en-boy oranı) ile büyük ölçüde çözülmüştür. Ayrıca, mentör ile birlikte fark edilen bir kimlik (ID) kayması sorunu, ID atamanın gerçek bir çoklu nesne takibi (Multi-Object Tracking) ile aynı şey olmadığını göstermiştir. Önümüzdeki hafta YOLO tabanlı tespitin sağlamlaştırılması, MOT algoritmalarını (BoT-SORT vb.) araştırılması ve klasik CV ile derin öğrenme yaklaşımlarının karşılaştırılması planlanmaktadır.

## 2. Yapılan İşler

### Gün 1 — 06.07.2026: OpenCV Temelleri, Piksel Matematiği ve Renk Uzayları

- **Yapılan İşlem:** OpenCV'nin ne olduğu, ne olmadığı ve bir görüntünün bilgisayar için aslında sadece piksel renk değerlerinden oluşan devasa bir sayı matrisi (NumPy dizisi) olduğunu öğrenilmiştir.
- **Çalışmanın Amacı:** Kütüphaneyi kullanmadan önce "hangi problemde OpenCV kullanılır" mantığının oturtulması hedeflenmiştir: çözüm kameradan gelen piksellerin rengine, şekline, konumuna veya hareketine mi bağlı — cevap evetse OpenCV devreye girmektedir.
- **OpenCV nedir / ne değildir:** Temelde C/C++ ile yazılmış, gerçek zamanlı görüntü/video işleme için 2500'den fazla algoritma (SVM, KNN gibi klasik makine öğrenmesi teknikleri dahil) barındıran açık kaynaklı bir kütüphane olduğu incelenmiştir. Kendi başına semantik (anlamsal) çıkarım yapan bir yapay zeka olmadığı — yani "bu resimde araba var mı?" sorusunu OpenCV tek başına cevaplayamayacağı; bunun için arka planda YOLO gibi bir derin öğrenme modeli çalıştırılması gerektiği saptanmıştır. OpenCV'nin görevinin, o modele görüntüyü doğru boyutta, doğru renkte ve temizlenmiş şekilde "tertemiz bir tabak" olarak sunmak olduğu belirlenmiştir.
- **Piksel ve matris mantığı:** Görüntülerin uint8 (0-255 arası işaretsiz 8-bit tam sayı) veri tipinde tutulduğu; 0'ın kapkaranlık, 255'in bembeyaz demek olduğu tespit edilmiştir. `.shape` özelliğinin renkli bir görselde (Yükseklik, Genişlik, Kanal Sayısı) sırasıyla döndüğü incelenmiş olup — örn. Full HD bir görsel için (1080, 1920, 3). Grayscale'e çevrilince kanal boyutu düşer, (1080, 1920) kalır. Bu yüzden boyutlandırma/kırpma işlemlerinde daima önce Y (satır/yükseklik), sonra X (sütun/genişlik) verilir (`img[y1:y2, x1:x2]`).
- **Renk uzayı dönüşümleri ve "Matplotlib krizi":** OpenCV görüntüyü varsayılan olarak BGR (Mavi-Yeşil-Kırmızı) sırasıyla okuduğu, ama Matplotlib'in RGB beklediği; dönüşüm yapılmazsa kırmızı nesnelerin mavi göründüğü saptanmıştır. Çözüm: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`. Grayscale'e geçişin işlem yükünü azalttığı ve formülünün `Y = 0.299·R + 0.587·G + 0.114·B` şeklinde olduğu (insan gözü yeşile daha duyarlı olduğu için G katsayısı en yüksek) öğrenilmiştir.
- **HSV'nin gücü:** BGR'nin ışık değişimlerine karşı hassas olduğu (3 kanal birbirine karışır); HSV (Hue: renk tonu, Saturation: doygunluk, Value: parlaklık) uzayının rengi ışıktan ayırdığı — gölge düştüğünde sadece V kanalının değiştiği, H (renk kimliği) kanalının sabit kaldığı öğrenilmiştir. Bu yüzden "kırmızı dubayı bul" gibi bir filtrelemenin, hava güneşli de olsa bulutlu da olsa çalıştığı doğrulanmıştır. Kritik detay: Fizikte Hue 360° iken, OpenCV'nin uint8 veri tipi (maks. 255) yüzünden H değeri 0–179 aralığına sıkıştırılmıştır; S ve V ise 0–255 arasındadır — filtreleme yaparken bu sınırlara dikkat edilmesi gerektiği anlaşılmıştır.
- **Temel geometrik dönüşümler:** `cv2.resize()` ile yeniden boyutlandırma (ör. kameradan gelen 1920×1080 görüntüyü YOLO'nun beklediği 640×640'a indirmek) ve interpolasyon mantığı (büyütme/küçültmede aradaki piksellerin değerine karar verme süreci) çalışılmış; NumPy dilimleme (slicing) ile ROI (Region of Interest — ilgilenilen bölge) kesme işlemleri uygulanmıştır.

### Gün 2 — 07.07.2026: Çizim, İnteraktif Arayüzler ve Nesne Tabanlı Programlama

- **Yapılan İşlem:** Koordinat sistemi, BGR renk tanımları, canlı video yakalama döngüsü, overlay/trackbar/mouse-callback ile interaktif pencereler ve temel şekil çizimi (çizgi, dikdörtgen) üzerinde çalışılmıştır.
- **Çalışmanın Amacı:** İleride kurulacak tespit sistemlerinde kullanıcıya görsel geri bildirim (kutu, yazı, renk göstergesi) verilmesi gerekeceği; bu yüzden çizim ve arayüz fonksiyonlarına hakim olunması hedeflenmiştir.
- **Koordinat sistemi:** OpenCV'de resmin sol üst köşesinin (0,0) referans noktası olduğu; X'in soldan sağa, Y'nin yukarıdan aşağıya arttığı ve en büyük değerlerin sağ alt köşede olduğu belirlenmiştir.
- **Video akış döngüsü (standart kalıp):** `cap = cv.VideoCapture(0)` → döngü içinde `ret, frame = cap.read()` → işle → `cv.imshow()` → `cv.waitKey()` → çıkışta `cap.release()` mimarisi incelenmiştir.
- **İnteraktif bileşenler:** `cv.displayOverlay()` ile pencere üzerine geçici bilgi yazma, `cv.createTrackbar()` / `cv.getTrackbarPos()` ile kaydırmalı parametre kontrolü (ör. üç trackbar ile canlı RGB renk karıştırma), `cv.setMouseCallback()` ile fare olaylarını (tıklama/sürükleme/bırakma) yakalama mekanizmaları oluşturulmuştur.
- **Çizim fonksiyonları:** `cv.line()`, `cv.rectangle()` fonksiyonları kullanılmış; kalınlık negatif veya `cv.FILLED` verilirse şeklin içinin dolu çizildiği görülmüştür. Renklerin BGR sırasıyla tanımlandığı (ör. `RED = (0,0,255)`, kırmızı en sonda!) tespit edilmiştir. Trackbar değerleri her zaman 0'dan başladığı için çizgi kalınlığının en az 1 olması `max(1, x)` ile garantilenmiştir.
- **Nesne tabanlı programlama (OOP):** `class` yapısının bir kalıp, `__init__`'in kurucu metod, `self`'in nesnenin kendisi olduğu incelenmiştir. Bu yapıyla, `q` tuşuna basınca güvenli şekilde kapanan basit bir pencere uygulaması (`App().run()`) kurulmuştur.

### Gün 3 — 08.07.2026: Renk Tabanlı Drone Tespiti ve Takibi (Uygulama Projesi)

- **Yapılan İşlem:** Gökyüzündeki bir drone'u HSV maskeleme + kontur tespiti ile tespit edip etrafına kutu çizen ve görünürlük süresini hesaplayan bir video işleme pipeline'ı yazılmıştır (Google Colab üzerinde, videoyu Drive'dan okuyup işlenmiş hâliyle geri kaydederek).
- **Çalışmanın Amacı:** Kameradan gelen ham görüntüden anlamlı bir nesne tespiti çıkarılması sürecinin uçtan uca (okuma → maskeleme → morfolojik işlem → kontur → kutu çizme → süre hesabı) deneyimlenmesi amaçlanmıştır.

**Kavram: Thresholding (Eşikleme) ve Masking (Maskeleme) Nedir?**

**Thresholding (Eşikleme):** Görüntüdeki pikselleri belirlenen bir renk/parlaklık aralığına göre ikiye ayırma işlemidir. Aralığa giren pikseller beyaza (255), girmeyenler siyaha (0) çevrilerek sonuçta ikili bir görüntü elde edilir. `cv.inRange(hsv, alt_sinir, ust_sinir)` fonksiyonu ile uygulanır.

**Masking (Maskeleme):** Thresholding sonucu oluşan bu siyah-beyaz haritayı (maske) orijinal görüntüye uygulayarak sadece ilgilenilen bölgeyi görünür bırakma, geri kalanını karartma işlemidir. `cv.bitwise_and(img, img, mask=maske)` ile yapılarak maskenin beyaz olduğu yerde orijinal görüntünün görünür kalması sağlanır.

Drone projesinde bu iki yöntem birlikte kullanılmıştır: önce HSV'de renk/parlaklık aralığına göre bir maske (thresholding) çıkarılmış, sonra bu maskedeki beyaz bölgelerin sınırları (`cv.findContours`) bulunarak drone'un konumu tespit edilmiştir.

- **Problem 1 — Parçalanmış maske:** Güneş açısı yüzünden drone'un bir tarafı parlak beyaz, gölgedeki tarafı ise grimsi kaldığı için maskenin tek parça değil 2-3 parça halinde çıktığı görülmüş; koddaki `max(konturlar)` ifadesinin sadece en büyük parçayı (sol tarafı) alıp diğerlerini yok saydığı saptanmıştır. Çözüm: V (parlaklık) alt sınırı 180'den 120'ye düşürülerek gölgeli grileri de dahil edilmiş; `cv.dilate` (morfolojik genişletme) ile beyaz alanlar dışa doğru şişirilip aradaki boşluklar doldurarak drone tek bir bütün bloğa dönüştürülmüştür.
- **Problem 2 — Lens parlaması / yanlış varsayım:** Parlak gökyüzüne bakan kamerada drone'un piksellerinin "simsiyah" değil, aslında V≈130-150 aralığında (grimsi/mavimsi, daha aydınlık) olduğu fark edilememiş; ilk eşiğin (`ust_siyah = [180,255,90]`) bu pikselleri de bulut sayıp tamamen maskeden sildiği, kontur bulunamadığı gözlenmiştir. Çözüm: Piksel değerleri yeniden incelenip V üst sınırı 150'ye çıkarılarak eşik gerçek verilere göre kalibre edilmiştir.
- **Zaman/süre hesaplama:** Video işlenirken Python'ın gerçek saatinin (`time.time()`) kullanılamayacağı öğrenilmiş — Colab'ın işleme hızı ile, videonun gerçek akış hızı aynı olmadığından FPS (saniyedeki kare sayısı) tabanlı kare sayımı kullanılması gerektiği anlaşılmıştır. İlk versiyonda sayacın sadece drone tespit edildiğinde arttığı ("kesintili kronometre" — drone gölgeye girince sayaç durmaktaydı); geliştirilen ikinci versiyonda ise sayacın her karede şartsız arttığı ve `ilk_tespit_kare` referans alınarak, drone maskeden anlık kaçsa bile videonun gerçek zaman çizelgesine sadık kalan doğru bir sürenin hesaplandığı görülmüştür:

```python
toplam_kare_sayisi += 1
hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
maske = cv.inRange(hsv_frame, alt_beyaz, ust_beyaz)
maske = cv.dilate(maske, cekirdek, iterations=2)
konturlar, _ = cv.findContours(maske, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

if cv.contourArea(en_buyuk_kontur) > 500:
    if ilk_tespit_kare is None:
        ilk_tespit_kare = toplam_kare_sayisi
    gecen_sure = (toplam_kare_sayisi - ilk_tespit_kare) / fps
```

- **Problem 3 — Şehir arka planında yanlış-pozitifler:** Drone alçalıp şehir manzarasına girdiğinde bina camları, yoldaki araçlar ve ağaç gölgeleri de aynı "siyah/koyu" eşiğine takılıp drone sanıldı; `cv.dilate` bu küçük yanlış lekeleri birleştirip ekranı kaplayan devasa bloklara dönüştürdü. Çözüm — Akıllı Geometrik Filtreleme (Heuristics): Sadece renge güvenmek yerine drone'un şekil özelliklerini de koda öğrettim: alan aralığı (çok küçük gürültüleri ve çok büyük gölge bloklarını eleyen 8000 < alan < 80000 sınırı) ve en-boy oranı (drone'un kolları açık olduğu için genelde yükseklikten daha geniş olmasından yola çıkan 1.1 < oran < 3.5 sınırı — böylece kare/dikey pencere gibi yanlış adaylar elendi):

```python
if 8000 < alan < 80000:
    oran = float(genislik_kutu) / float(yukseklik_kutu)
    if 1.1 < oran < 3.5:
        # geçerli drone adayı olarak işaretle
```

- **Ek konular:** Aynı gün içinde afin dönüşümleri (öteleme/warpAffine ile kaydırma, `getRotationMatrix2D` ile döndürme ve ölçekleme), dairesel maskeleme (`cv.circle` + `cv.bitwise_and`) ve histogram analizini de çalıştım. Histogram, bir resimdeki her parlaklık/renk değerinden kaç piksel olduğunu gösteren grafik (`cv.calcHist`) — resmin karanlık mı aydınlık mı, kontrastlı mı olduğunu anlamamı sağlıyor; renkli görüntülerde B/G/R kanallarını `cv.split` ile ayırıp her birini ayrı eğri olarak çizdim.

### Gün 4 — 09.07.2026: YOLO ile Derin Öğrenme Tabanlı İnsan Tespiti ve Takibi

- **Yapılan İşlem:** Bir kafe güvenlik kamerası videosunda YOLO modeli ile COCO veri setindeki "person" (sınıf ID 0) nesnesi tespit edilmiş; her kişiye kalıcı bir kimlik numarası atamak için `model.track(..., persist=True)` kullanılmıştır.
- **Çalışmanın Amacı:** Klasik CV'nin (Gün 3'teki renk/kontur tabanlı yöntem) karmaşık, çok nesneli ve semantik ayrım gerektiren ("bu bir insan mı, gölge mi?") sahnelerde yetersiz kaldığı yerde derin öğrenme tabanlı tespitin nasıl çalıştığının görülmesi amaçlanmıştır.
- **Model boyutu seçimi:** YOLO'nun n (nano — en hızlı, mobil/edge cihazlar için), s (small — laptop ve orta seviye GPU'lar için genel varsayılan), m (medium — doğruluğun öncelikli olduğu sunucu tarafı varsayılan), l/x (large/xlarge — her mAP puanının önemli olduğu, güçlü GPU gerektiren) varyantları arasındaki hız–doğruluk dengesi karşılaştırılmıştır.
- **NMS (Non-Maximum Suppression):** Modelin bazen aynı kişinin üzerine hafif kaymış 2-3 farklı çerçeve çizdiği görülmüştür. NMS algoritmasının, üst üste binen çerçevelerden güven skoru (confidence) en yüksek olanını seçip diğerlerini elediği saptanmıştır. Bu durum `iou` (Intersection over Union) parametresiyle kontrol edilmiştir: `iou=0.4` değerinin, "iki çerçeve %40'tan fazla üst üste biniyorsa düşük skorlu olanı sil" anlamına geldiği test edilmiştir.
- **Her kişiye ID atamak için kullanılan kod kesiti:**

```python
results = model.track(frame, classes=[0], conf=0.3, persist=True, iou=0.4)
annotative_frame = frame.copy()

if results[0].boxes.id is not None:
    boxes = results[0].boxes.xyxy.int().cpu().tolist()
    ids = results[0].boxes.id.int().cpu().tolist()
    for box, track_id in zip(boxes, ids):
        x1, y1, x2, y2 = box
        cv.rectangle(annotative_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        etiket = f"Target {track_id}"
        cv.putText(annotative_frame, etiket, (x1, y1 - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
```

- **Mentör geri bildirimi ve fark edilen eksik:** Video mentörle birlikte izlerken, bir kişinin önüne başka biri geçip (occlusion / kapanma) tekrar ortaya çıktığında ID numarasının değiştiği fark edilmiştir (ör. ID:5 olan kişi, ID:9 olarak yeniden atanıyordu). Mentör tarafından bunun nasıl önlenebileceği sorulmuş; o an bunun ayrı bir problem (Multi-Object Tracking) olduğunun düşünülmediği ve `persist=True` parametresinin zaten "tracking" yaptığı, dolayısıyla bunu çözdüğü şeklinde cevap verilmiştir.
- Mentör tarafından bu yaklaşımın yetersiz olduğu, kapanma sonrası kimliği doğru şekilde koruyabilmek için BoT-SORT gibi özel Multi-Object Tracking (MOT) algoritmalarının kullanıldığı belirtilmiştir. `persist=True` parametresinin sadece kare-kareye eşleştirmeyi hafızada tuttuğu; nesne birkaç kare boyunca tamamen kaybolduğunda (kapandığında) bu eşleştirmenin kırılabildiği anlaşılmıştır.
- **Sonraki adım:** Bu konuyu henüz araştırılmamış olup; 5 günlük raporu tamamladıktan sonra detaylı incelemeyi planlıyorum.
- **İkinci bir insan tespiti denemesi (Üsküdar kamerası):** Aynı YOLO mantığı, çok daha uzak ve geniş açılı bir kamera görüntüsü (Üsküdar'da çekilmiş bir video) üzerinde de denenmiştir. Kafe videosundaki kameraya göre insanlar ekranda çok daha küçük kaldığı için modelin tespitte zorlanmaması adına `imgsz=1024` parametresini eklenmiştir (görüntü modele daha yüksek çözünürlükte verilerek küçük nesnelerin de yakalanması sağlanmıştır); kafe videosunda mesafe yakın olduğu için bu parametreye ihtiyaç duyulmamıştır. İki notebook (`people_in_cafe.ipynb` ve `people_in_uskudar.ipynb`) arasındaki temel fark bu olmuştur:

```python
# people_in_cafe.ipynb (yakın çekim kamera)
results = model(frame, classes=[0], conf=0.5, iou=0.4)

# people_in_uskudar.ipynb (uzak / geniş açılı kamera)
results = model(frame, classes=[0], conf=0.5, imgsz=1024)
```

- **Öğrenilen:** `imgsz` parametresinin modele verilen görüntünün işlenme çözünürlüğünü belirlediği; kamera uzaklaştıkça ve insanlar küçüldükçe bu değeri yükseltmenin tespit performansını doğrudan etkilediği — yani doğru parametreyi seçmenin sahne/kamera koşuluna (mesafe) bağlı olduğu öğrenilmiştir.
- **Araç/ortam tarafında öğrenilenler:** Bu süreçte Google Colab'ı (dosya/Drive bağlama, GPU çalışma zamanı, hücre bazlı çalıştırma — kurulum gerektirmeden tarayıcı üzerinden çalışan bir Jupyter Notebook ortamı) etkin şekilde kullanılması öğrenilmiş; ayrıca Colab ortamını VS Code'a bağlayarak kodun kendi editöründe yazılıp Colab'ın işlem gücünde çalıştırılması da öğrenilmiştir.

### Gün 5 — 10.07.2026: Nesne Tespit Algoritmaları Teorisi, YOLO Mimarisi ve SAHI Entegrasyonu

**Yapılan İşlem:** YOLO (You Only Look Once) algoritmasının arka plandaki çalışma mantığı, mimarisi ve diğer nesne tespit algoritmalarından farkı teorik olarak araştırılmıştır. Ardından, Üsküdar kamerasındaki uzak mesafe insan tespiti problemine (çok küçük piksel alanına sahip nesnelerin gözden kaçması) çözüm olarak SAHI algoritmasının YOLO ile entegrasyon mantığı incelenmiştir.

**Çalışmanın Amacı:** Bir yapay zeka modelini sadece "kara kutu" (black-box) olarak kullanmak yerine; onun görüntüyü nasıl işlediğinin, matematiksel olarak nasıl tahmin yaptığının, mimari sınırlarının ve neden küçük nesnelerde (Üsküdar videosu) zorlandığının tam olarak kavranması hedeflenmiştir.

## 3. Öğrenilen Kavramlar ve Teorik Analizler:

### 3.1. Nesne Tespit Algoritmalarına Genel Bakış

Literatürdeki nesne tespit algoritmalarının temel olarak ikiye ayrıldığı öğrenilmiştir:

- **İki Aşamalı (Two-Stage) Algoritmalar (Örn: Faster R-CNN):** Önce görüntüde nesne olabilecek potansiyel bölgeleri (Region Proposals) çıkarır, ardından bu bölgeleri sınıflandırır. Doğruluk (Accuracy) oranları çok yüksektir ancak çalışma süreleri yavaş olduğu için gerçek zamanlı (real-time) projelere uygun değillerdir.
- **Tek Aşamalı (One-Stage) Algoritmalar (Örn: YOLO, SSD):** Görüntüye sadece bir kez bakar ve tüm sınıflandırma ile kutu (bounding box) tahminlerini tek bir sinir ağı geçişinde (single forward pass) yapar. Doğruluk ile inanılmaz bir hız dengesi sunarak gerçek zamanlı tespitin endüstri standardı olmuşlardır.

### 3.2. YOLO Nedir ve Nerelerde Kullanılır?

YOLO (You Only Look Once), tek aşamalı nesne tespiti yaklaşımının en popüler ve güçlü temsilcisidir. Görüntüyü kare kare taramak yerine tüm fotoğrafa global bir bağlamda bakar.

- **Kullanım Alanları:** Milisaniyelerin hayati önem taşıdığı her yerde kullanılır. Otonom araçlar (yaya ve şerit takibi), İHA/İDA (insansız deniz/hava araçları) sistemleri, güvenlik kameralarında anomali tespiti, endüstriyel üretim hatlarında hata denetimi ve sağlık sektöründe anlık doku analizi gibi alanlarda sektör standartıdır.

### 3.3. YOLO Nasıl Çalışır ve Mimarisi Nasıldır?

YOLO, kendisine verilen görüntüyü alır ve S×S boyutlarında bir ızgaraya (grid) böler. Eğer bir nesnenin merkez noktası belirli bir ızgara hücresine düşerse, o nesneyi tespit etmekten tamamen o hücre sorumlu olur. YOLO mimarisi üç ana omurgadan oluşur:

- **Backbone (Omurga):** Kameradan gelen görüntüden temel özellikleri (kenarlar, renkler, dokular) çıkaran derin sinir ağı katmanıdır (Genellikle CSPDarknet mimarisi kullanılır).
- **Neck (Boyun):** Backbone'dan gelen farklı boyutlardaki özellikleri birleştiren kısımdır. Görüntüdeki hem çok büyük hem de çok küçük nesnelerin özelliklerini harmanlayarak (Feature Pyramid Network mantığıyla) ağın vizyonunu genişletir.
- **Head (Baş/Çıktı):** İşlenmiş veriyi alıp son kararı veren kısımdır. Nesnenin kutu koordinatlarını (x, y, genişlik, yükseklik), o kutunun içinde gerçekten bir nesne olma olasılığını (confidence score) ve nesnenin sınıfını (person, car vb.) tahmin eder.

### 3.4. SAHI (Slicing Aided Hyper Inference) ile "Model Körlüğünü" Aşmak

YOLO'nun mimarisi öğrenildikten sonra Üsküdar videosundaki zayıf noktası tespit edilmiştir. 4K gibi yüksek çözünürlüklü bir görüntü modele verildiğinde, modelin bunu kendi işleyebileceği boyuta (örn: 1024x1024) sıkıştırdığı görülmüştür. Bu sıkıştırma esnasında en arkadaki 10x10 piksellik bir insanın, 2x2 piksellik anlamsız bir lekeye dönüştüğü saptanmıştır (Model Körlüğü). Sıkıştırmadan verilmesi halinde VRAM'in yetmediği ve sistemin Out of Memory (OOM) hatası verdiği anlaşılmıştır. Bu sorunu çözmek için SAHI mantığı incelenmiştir:

- **Dilimleme (Slicing):** Orijinal büyük görüntü küçültülmez; belirli boyutlarda (örneğin 512x512) birbirinin üzerine binen (overlapping) ızgaralara bölünür.
- **Bağımsız Çıkarım (Inference):** Her bir dilim, ayrı birer fotoğrafmış gibi YOLO modeline beslenir. Uzaktaki insan, bu küçük dilimin içinde net bir orana sahip olduğu için model onu rahatlıkla bulur.
- **Birleştirme (Merging):** Tüm dilimlerdeki tespit sonuçları, orijinal görüntünün koordinat sistemine haritalanır. Aynı insanın farklı dilimlerde iki kez tespit edildiği kesişim bölgeleri, NMS (Non-Maximum Suppression) algoritmasıyla temizlenerek tek kutuya düşürülür.

**Sonuç:** Bir modelin arka plandaki mimarisini (Backbone, Neck, Head) ve görüntüyü ızgaralara bölme mantığını bilmenin; karşılaşılan limitasyonlarda (uzak mesafe körlüğü) SAHI gibi "görüntü dilimleme" algoritmalarının neden kesin çözüm olduğunu kavramayı sağladığı görülmüştür. SAHI entegrasyonunun model değiştirmeden başarıyı artırdığı, ancak işlenecek resim sayısı arttığı için FPS'i (hızı) düşüren bir trade-off (ödünleşim) yarattığı analiz edilmiştir.

## 4. Karşılaşılan Zorluklar / Öğrenilenler

Bu hafta karşılaşılan teknik zorluklar ve bunlardan çıkarılan dersler özetlenmektedir (detaylı anlatımları ilgili gün başlıkları altında yer almaktadır):

| Yaşanan Zorluk | Çözüm / Öğrenilen Ders |
|---|---|
| Drone'un güneşten aydınlık ve gölgede kalan yüzeylerinin farklı parlaklıkta olması nedeniyle maskenin 2-3 parçaya bölünmesi. | HSV'de V (parlaklık) alt sınırı 180'den 120'ye düşürülmüş; parçaları birleştirmek için `cv.dilate` morfolojik genişletmesi uygulanmıştır. |
| Lens parlaması nedeniyle drone piksellerinin aslında V≈130-150 aralığında olup ilk eşikte (V<90) tamamen silinmesi. | Piksel değerleri yeniden incelenip üst V sınırı 150'ye çıkarılarak eşik gerçek verilere göre kalibre edilmiştir. |
| Drone şehir manzarasına indiğinde bina camları, araçlar ve ağaç gölgelerinin de siyah eşiğine takılıp yanlış-pozitif üretmesi. | Sadece renk yetmemiş; alan aralığı (8000<alan<80000) ve en-boy oranı (1.1 <oran< 3.5) filtreleri eklenerek geometrik heuristik kurulmuştur. |
| Süre hesaplarken Python'ın gerçek saatinin (`time.time()` / `datetime`) kullanılması; Colab işleme hızının videonun gerçek akışıyla örtüşmemesi nedeniyle saniyelerin tutmaması. | Gerçek zaman yerine FPS tabanlı kare sayımına geçilmiş; `ilk_tespit_kare` referansıyla videonun kendi zaman çizelgesine sadık kalarak doğru süreyi hesaplanmıştır. |
| Üsküdar videosunda kamera çok uzak olduğu için insanların ekranda küçük kalması ve YOLO'nun varsayılan ayarlarla tespit edememesi. | Modele verilen görüntü çözünürlüğü `imgsz=1024` ile yükseltilerek küçük nesnelerin de yakalanması sağlanmıştır. |
| Kapanma (occlusion) sonrası bir kişinin ID'sinin değişmesi — `persist=True` kullanımına rağmen bunun gerçek bir tracking çözümü olduğunun fark edilememesi. | Mentör yönlendirmesiyle bunun ayrı bir problem (Multi-Object Tracking) olduğu öğrenilmiş; BoT-SORT gibi algoritmaların araştırılması planlanmıştır. |

## 5. Sorular / İhtiyaç Duyduğum Destek

- Çoklu drone senaryosunda geometrik filtreleme (alan + en-boy oranı) hâlâ %100 stabil değildir; bu konuda parametre ayarlama dışında önerilebilecek bir yöntem (ör. background subtraction) var mıdır?
- YOLO modelinin bu proje kapsamında sıfırdan eğitilmesi mi yoksa hazır COCO ağırlıklarıyla mı ilerlemesi gerektiği; hedef kullanım senaryosuna göre hangisinin daha uygun olduğu tartışılmalıdır. Eğer projemiz özelinde modelin sıfırdan eğitilmesi (fine-tuning) gerekiyorsa, bu sürecin adımları nasıl planlanmalıdır?
- Renk tabanlı takip ile YOLO tabanlı takibi hibrit şekilde birleştirmenin (ör. YOLO ile genel tespit + renk ile hızlı doğrulama) mantıklı bir yön mü olduğu, yoksa gereksiz karmaşıklık mı katacağı değerlendirilmelidir?
- BoT-SORT dışında incelenmesi önerilen başka bir MOT algoritması var mıdır (ör. ByteTrack, DeepSORT)? Projemizin ihtiyacına göre hangisiyle başlamak daha doğru olacaktır?

## 6. Gelecek Hafta Planı

- YOLO tabanlı insan takibini (ID atama + NMS) daha fazla test videosu üzerinde denenip model boyutları (n/s/m/l) arasında performans karşılaştırmasının yapılması.
- Multi-Object Tracking (MOT) algoritmalarının teorik altyapısının öğrenilmesi, öğrenilen Multi-Object Tracking (MOT) algoritmalarının (BoT-SORT vb.) insan ve araç takibi projelerindeki test videoları üzerinde uygulanarak performanslarının incelenmesi. Occlusion sonrası ID kayıplarını önleyip önlemediğini test edilmesi.
- Özel veri setleri oluşturularak YOLO tabanlı model etiketleme ve sıfırdan eğitme (fine-tuning) süreçlerine giriş yapılması.
- Teorik analizi yapılan SAHI (Slicing Aided Hyper Inference) algoritmasının, uzak mesafe hedef problemi olan Üsküdar videosu üzerinde kodlanarak pratik edilmesi.
- Haar Cascades yöntemi kullanılarak yüz tespiti (face detection) algoritmalarının incelenmesi ve tespit edilen yüzlerin kutucuklarla (bounding box) görselleştirilmesi.
- Renk histogramları, histogram eşleme (histogram matching) ve şablon eşleştirme (template matching) gibi görüntü eşleştirme kavramlarının araştırılarak uygulamalı olarak test edilmesi.
- Edinilen bilgisayarlı görü ve nesne takibi (MOT) yetkinliklerinin, mentör onayı ve uygunluğu doğrultusunda şirketin aktif projelerinden birine entegre edilerek geliştirici ekibe aktif katkı sağlaması.
