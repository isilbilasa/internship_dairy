**STAJ 2. HAFTA RAPORU**

Işıl Bilasa

Hafta 2

**Konu: YOLO ile Çoklu Nesne Takibi, Model Eğitimi ve Klasik Görüntü İşleme Teknikleri**

13 – 17 Temmuz 2026

# İçindekiler

[İçindekiler [2](#içindekiler)](#içindekiler)

[1. Kısa Özet [3](#_Toc235230945)](#_Toc235230945)

[2. Yapılan İşler [3](#_Toc235230946)](#_Toc235230946)

[Gün 6 --- 13.07.2026: YOLO ile Çoklu Nesne Takibi (MOT) ve SAHI ile Küçük Nesne Tespiti [3](#_Toc235230947)](#_Toc235230947)

[Gün 7 --- 14.07.2026: Label Studio ile Veri Etiketleme ve YOLO Eğitim Altyapısının Hazırlanması [6](#_Toc235230948)](#_Toc235230948)

[Gün 8 --- 16.07.2026: Model Eğitiminin Tamamlanması, Metriklerin Yorumlanması ve Gerçek Zamanlı Kişi Sayımı [7](#_Toc235230949)](#_Toc235230949)

[Gün 9 --- 17.07.2026: Klasik Görüntü Ön İşleme Teknikleri ve Haar Cascade ile Yüz Tespiti [10](#_Toc235230950)](#_Toc235230950)

[3. Karşılaşılan Zorluklar / Öğrenilenler [13](#karşılaşılan-zorluklar-öğrenilenler)](#karşılaşılan-zorluklar-öğrenilenler)

<span id="_Toc235230945" class="anchor"></span>1. Kısa Özet

Hafta 2 kapsamında, Hafta 1'de tespit edilen kimlik (ID) kayması probleminden yola çıkarak Ultralytics YOLO'nun çoklu nesne takip (Multi-Object Tracking) altyapısı ayrıntılı olarak incelenmiş; ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, FastTracker ve TrackTrack algoritmaları teknik açıdan karşılaştırılarak kalabalık sahnelerde kimlik sürekliliği açısından Deep OC-SORT ve TrackTrack algoritmalarının öne çıktığı değerlendirilmiştir. Uzak mesafedeki küçük nesnelerin tespitini iyileştirmek amacıyla SAHI (Slicing Aided Hyper Inference) algoritması araştırılmış ve örnek uygulaması gerçekleştirilmiştir. Ardından özel bir insan tespit modeli eğitmek için market_video.mp4 videosu karelere ayrıştırılmış, Label Studio ile tek sınıflı olarak etiketlenmiş, %80/%10/%10 oranında train/val/test klasörlerine dağıtılmış ve data.yaml dosyası yapılandırılmıştır. Google Colab üzerinde yolov8n.pt mimarisiyle 50 epoch boyunca gerçekleştirilen eğitim sonucunda yüksek başarı metrikleri elde edilmiştir (Precision 0,948; Recall 0,935; mAP50 0,959; mAP50-95 0,601); eğitim logları ve grafikleri (F1, PR eğrisi, Confusion Matrix, results.png vb.) overfitting belirtisi göstermediği yönünde yorumlanmıştır. Eğitilen model, ByteTrack ve LineZone bileşenleri kullanılarak markette gerçek zamanlı giriş-çıkış/kişi sayımı uygulamasında test edilmiştir. Haftanın son gününde ise derin öğrenme öncesi klasik bilgisayarlı görü hattı bütünsel olarak ele alınmış; konvolüsyon ve kernel kavramları, filtreleme/bulanıklaştırma yöntemleri (Box, Gaussian, Median, Bilateral), Sobel ve Canny ile kenar tespiti, Otsu eşikleme, kontur tespiti, histogram/şablon eşleştirme teknikleri ve Haar Cascade tabanlı yüz tespiti incelenerek HSV tabanlı kırmızı top takibi uygulamasıyla pekiştirilmiştir. Önümüzdeki hafta, seçilen takip algoritmalarının test videoları üzerinde karşılaştırmalı olarak denenmesi ve model eğitim/etiketleme sürecinin genişletilmesi planlanmaktadır.

<span id="_Toc235230946" class="anchor"></span>2. Yapılan İşler

<span id="_Toc235230947" class="anchor"></span>Gün 6 --- 13.07.2026: YOLO ile Çoklu Nesne Takibi (MOT) ve SAHI ile Küçük Nesne Tespiti

- **Yapılan İşlem:** YOLO ile Çoklu Nesne Takibi (MOT) ve SAHI ile Küçük Nesne Tespiti**Çalışmanın Amacı:** Hafta 1'de fark edilen ID kayması probleminin çözümü için kalabalık sahnelerde kullanılabilecek en uygun takip yaklaşımının belirlenmesi ve ilerleyen aşamalarda geliştirilecek özel YOLO modelinin daha yüksek doğruluk ve daha kararlı takip performansı elde etmesine yönelik bir ön hazırlık yapılması hedeflenmiştir.

- **Nesne takibi (object tracking) kavramı:** Bir nesnenin yalnızca ilgili karede tespit edilmesinin ötesine geçilerek, videonun sonraki karelerinde de aynı nesnenin benzersiz bir kimlik (ID) ile tanınmasının sağlandığı işlem olarak tanımlanmıştır. Nesne algılamadan (detection) farkının, algılamanın her karede bağımsız çalışıp nesnenin kimliğini bilmemesine karşın takibin nesneyi bir önceki karedeki kimliğiyle ilişkilendirmesi olduğu belirlenmiştir.

- **Ultralytics YOLO'nun avantajları:** Video akışlarının doğruluktan ödün vermeden gerçek zamanlı işlenebilmesi (verimlilik), birden fazla takip algoritması ve konfigürasyonunun desteklenmesi (esneklik), basit Python API/CLI ile hızlı entegrasyon (kullanım kolaylığı) ve özel eğitilmiş modellerle uyumlu çalışabilmesi (özelleştirilebilirlik) incelenmiştir. Gerçek dünya uygulama alanı olarak ulaşım sektöründe araç takibi, perakendede insan takibi ve akuakültürde balık takibi örnek olarak değerlendirilmiştir.

- **Temel çalışma mantığı:** Takip işleminin “Tahmin Et -- Ölç -- Güncelle” döngüsü üzerinden ilerlediği; bir nesnenin bir sonraki karedeki konumunun Kalman Filtresi ile tahmin edildiği, ardından gerçek konumun ölçülüp tahminle karşılaştırılarak aynı kimliğin atanıp atanmayacağına karar verildiği öğrenilmiştir. persist=True parametresinin sistemin videonun başından itibaren gördüğü kimlikleri hafızasında tutmasını sağladığı, persist=False durumunda ise her karenin ilk kez görülüyormuş gibi işlenerek takip sürekliliğinin bozulduğu tespit edilmiştir.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p>from ultralytics import YOLO</p>
<p>model = YOLO("yolo26n.pt")</p>
<p># Varsayılan takipçi (BoT-SORT)</p>
<p>results = model.track(source="video_kaynağı", show=True)</p>
<p># ByteTrack'e geçiş</p>
<p>results = model.track(source="video_kaynağı", show=True, tracker="bytetrack.yaml")</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

- **Desteklenen altı takip algoritması:** Her algoritmanın ilgili YAML konfigürasyon dosyası tracker parametresine verilerek etkinleştirildiği belirlenmiştir. Karşılaştırma aşağıdaki tabloda özetlenmiştir:

| **Algoritma** | **Çalışma Mantığı**                                                                                                                                | **En Uygun Kullanım Alanı**                                                           |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| ByteTrack     | Yalnızca konum/hız bilgisine dayalı Kalman Filtresi kullanır; görünüm modeli ve kamera hareketi dengelemesi içermez.                               | Sabit kameralar, trafik izleme, düşük işlem yükü gerektiren gerçek zamanlı sistemler. |
| BoT-SORT      | ByteTrack'e kamera hareketi dengeleme ve isteğe bağlı ReID ekler; varsayılan takipçidir.                                                           | Hareketli kameralar (drone, araç içi), kalabalık ortamlar.                            |
| OC-SORT       | Gözlem odaklı düzeltmelerle doğrusal olmayan hareketleri (ani yön değişimleri) ReID maliyeti olmadan daha iyi çözer.                               | Spor müsabakaları, dans gibi düzensiz hareketli sahneler.                             |
| Deep OC-SORT  | OC-SORT'un hareket tahmin gücüne uyarlanabilir görünüm birleştirme (ReID) ve kamera hareketi dengelemesi ekler.                                    | Kimlik karışmalarının yoğun yaşandığı kalabalık veya hareketli kamera sahneleri.      |
| FastTracker   | Görünüm modeli içermeyen, tıkanıklık (occlusion) farkındalığına sahip bir ByteTrack varyantıdır; Kalman geri alma mekanizması kullanır.            | Sık kısmi örtüşmelerin yaşandığı gerçek zamanlı, tespit odaklı sistemler.             |
| TrackTrack    | Çoklu ipucu (HMIoU, ReID mesafesi, güven ve açı bilgisi) kullanarak iz perspektifine dayalı ilişkilendirme yapar; kopya kimlik oluşumunu bastırır. | Kopya kimlik sorununun yoğun yaşandığı, sık tıkanıklıklı kalabalık sahneler.          |

- **Algoritma seçim yol haritası:** En hızlı/basit temel çözüm için ByteTrack; el kamerası/drone/hareketli kamera için BoT-SORT; doğrusal olmayan hareket ve ReID ihtiyacı yoksa OC-SORT; kalabalık ve hareketli kamerada kimlik değişimi ana sorunsa Deep OC-SORT veya TrackTrack; sık kısmi örtüşme yaşanan ve ReID bütçesi olmayan gerçek zamanlı sistemlerde FastTracker tercih edilmesi gerektiği sonucuna varılmıştır. Nesne takibinde yalnızca işlem hızının değil; kimlik sürekliliği, örtüşme başarısı, kamera hareketine uyum ve gerçek zamanlı performans kriterlerinin birlikte değerlendirilmesi gerektiği görülmüştür.

- **Re-ID (Yeniden Tanımlama) kavramı:** Nesneler bir sahnede birbirinin arkasından geçtiğinde (oklüzyon) sistemin nesnenin yeni mi yoksa daha önce görülmüş mü olduğunu ayırt etmesinin zorlaştığı; Re-ID mekanizmasının nesnenin yalnızca konumuna değil görsel özelliklerine de bakarak kimliğini korumasını sağladığı, böylece geçici olarak görüş alanından çıkan bir nesnenin tekrar belirdiğinde görünüm benzerliği üzerinden aynı kimlikle ilişkilendirilebildiği öğrenilmiştir.

- **Uygulamalı değerlendirme -- kalabalık kafe sahnesi:** İnsanların sürekli birbirinin önünden geçtiği kalabalık bir kafe videosunda en büyük risk faktörünün kimlik karışması (ID swapping) olduğu belirlenmiştir. Yalnızca konuma dayanan standart algoritmaların (ör. ByteTrack) ani duruş, yön değiştirme veya uzun süreli örtüşme durumlarında yetersiz kaldığı; bu nedenle Deep OC-SORT veya TrackTrack algoritmalarının bu tür sahneler için daha uygun olduğu sonucuna varılmıştır. İlgili konfigürasyonda with_reid: True ayarının yapılması, performans sorunu yaşanması hâlinde ise daha hafif BoT-SORT'a geçilebileceği not edilmiştir.

- **Takip algoritmalarına ait parametreler:** Tüm takipçiler tarafından paylaşılan ortak parametreler (track_high_thresh, track_low_thresh, new_track_thresh, track_buffer, match_thresh, fuse_score, gmc_method, proximity_thresh, appearance_thresh, with_reid, model vb.) incelenmiştir. BoT-SORT ve Deep OC-SORT'ta kalabalık sahnelerde ReID'in etkinleştirilip appearance_thresh değerinin yükseltilmesi (BoT-SORT için ~0,85+, Deep OC-SORT için 0,92-0,95), OC-SORT'ta delta_t/inertia/use_byte ile doğrusal olmayan harekete hassasiyet ayarı, FastTracker'da tıkanıklık yönetimine yönelik occ\_\* parametreleri, TrackTrack'te ise iou/reid/conf/angle ağırlıklarıyla çoklu ipucu ilişkilendirmesinin özelleştirilebildiği tespit edilmiştir. ReID özelliğinin işlem yükünü azaltmak için varsayılan olarak kapalı geldiği; model: auto seçeneğinin minimum ek yükle çalıştığı, dışa aktarılmış özel bir ReID modelinin ise daha ayırt edici fakat daha maliyetli sonuçlar sunduğu belirlenmiştir.

- **SAHI (Slicing Aided Hyper Inference) ile küçük nesne tespiti:** Standart tespit modellerinin görüntüyü sabit ve küçük bir çözünürlüğe (ör. 640x640) indirgemesi nedeniyle uzak mesafedeki nesnelerin birkaç piksele düşerek fark edilemediği; bu sorunu çözmek için SAHI'nin görüntüyü örtüşmeli dilimlere bölüp her dilimde ayrı çıkarım yaparak sonuçları birleştirdiği incelenmiştir. Uygulama adımları: (1) gerekli kütüphanelerin (sahi, ultralytics, opencv-python) kurulması, (2) SAHI'nin kendisinin bir model olmayıp YOLOv8 gibi modellerin üzerine oturan bir çıkarım stratejisi olması nedeniyle en yüksek doğruluk için yolov8x modelinin tercih edilmesi, (3) videonun kare kare okunup her karenin dilimlere bölünmesi, her dilimin modele verilmesi ve sonuçların tek karede birleştirilmesi.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p>import cv2 as cv</p>
<p>from sahi.predict import get_sliced_prediction</p>
<p>from sahi import AutoDetectionModel</p>
<p>detection_model = AutoDetectionModel.from_pretrained(</p>
<p>model_type='ultralytics', model_path='yolov8x.pt',</p>
<p>confidence_threshold=0.3, device="cuda:0"</p>
<p>)</p>
<p>result = get_sliced_prediction(</p>
<p>frame, detection_model,</p>
<p>slice_height=512, slice_width=512,</p>
<p>overlap_height_ratio=0.2, overlap_width_ratio=0.2</p>
<p>)</p>
<p># result.object_prediction_list üzerinden kutular çizilip video olarak kaydedilir.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

- **SAHI ince ayar önerileri ve performans gözlemi:** slice_height/slice_width'in çok büyük tutulmasının (ör. 1024) yeterli yakınlaştırmayı engellediği, küçük nesneler için 512-640 aralığının uygun olduğu; overlap oranlarının 0,2-0,3 aralığında tutulmasının dilim sınırında nesne kaçırılmasını önlediği; GPU olmadan gerçek zamanlı işlemenin zor olduğu, GPU yoksa videonun kare kare işlenip yeni bir dosya olarak kaydedilmesinin daha uygun olduğu belirlenmiştir. 30 FPS'lik 14 saniyelik bir videoda 420 kare ve karede 15 dilim olması durumunda yaklaşık 6.300 ayrı çıkarım yapılması gerektiği hesaplanmış; yolov8x gibi ağır bir modelle bu durumun GPU üzerinde ciddi işlem yükü oluşturduğu, doğruluk arttıkça işlem maliyetinin de belirgin şekilde yükseldiği gözlemlenmiştir.

- **Model eğitimi için video verisinin kareye ayrıştırılması:** Model eğitiminde kullanılacak veri setinin hazırlanması amacıyla OpenCV ile geliştirilen bir Python fonksiyonu kullanılarak videoların belirlenen frame_interval değerine göre kare kare okunduğu ve seçilen görüntülerin .jpg formatında kaydedildiği; bu görüntülerin sonraki aşamada Label Studio'da etiketlenerek YOLO veri kümesinin oluşturulmasında kullanılacağı belirtilmiştir.

**Gün 6 Özetle:** Ultralytics YOLO'nun çoklu nesne takip altyapısı ve altı takip algoritmasının çalışma prensipleri, kullanım alanları ve parametreleri karşılaştırılmış; kalabalık insan topluluklarında Deep OC-SORT ve TrackTrack'in daha uygun olduğu değerlendirilmiştir. SAHI algoritması araştırılmış ve örnek video üzerinde uygulanmış; ayrıca model eğitimine hazırlık kapsamında video verileri karelere ayrıştırılarak etiketlemeye uygun veri yapısı oluşturulmuştur.

<span id="_Toc235230948" class="anchor"></span>Gün 7 --- 14.07.2026: Label Studio ile Veri Etiketleme ve YOLO Eğitim Altyapısının Hazırlanması

- **Yapılan İşlem:** Bir önceki gün karelere ayrıştırılan market_video.mp4 videosu Label Studio kullanılarak etiketlenmiş, elde edilen veri seti YOLOv8 OBB formatında dışa aktarılmış ve model eğitimi için gerekli klasör yapısına (images/labels altında train, val, test) uygun şekilde yeniden düzenlenmiştir. Ayrıca data.yaml dosyasının yapısı, eğitim-doğrulama-test veri kümelerinin oluşturulma mantığı ve görüntü-etiket dosyalarının eşleştirilmesi incelenmiş; hazırlanan veri setiyle Google Colab'da model eğitimi başlatılmış ve karşılaşılan dosya yolu (path) hatası değerlendirilmiştir.

- **Çalışmanın Amacı:** İlerleyen aşamalarda gerçekleştirilecek özel YOLO modelinin sağlıklı şekilde eğitilebilmesi için gerekli, doğru etiketlenmiş ve doğru klasör/yol yapısına sahip bir veri hazırlama altyapısının oluşturulması hedeflenmiştir.

- **Label Studio ile etiketleme çalışması:** Kareler içerisindeki tüm insanların tek bir sınıf altında etiketlendiği, elde edilen veri setinin YOLOv8 OBB With Images formatında dışa aktarıldığı ve bu süreçte Label Studio arayüzü ve etiketleme iş akışına yönelik pratik deneyim kazanıldığı belirtilmiştir.

- **Veri setinin klasör yapısına göre düzenlenmesi:** Dışa aktarılan veri seti VS Code'da açılarak images ve labels klasörleri altında train/val/test alt klasörleri oluşturulmuş; fotoğraflar ve etiket dosyaları %80 train, %10 val ve %10 test oranında dağıtılmıştır. Google Colab'a aktarılan veri setiyle eğitim başlatılmış, ancak veri yollarının (path) tanımlanmasına ilişkin bir hata nedeniyle eğitim tamamlanamamış; bu tür hataların çoğunlukla data.yaml'daki yollar ile klasör yapısı arasındaki uyumsuzluktan kaynaklandığı değerlendirilmiştir.

- **data.yaml dosyasının işlevi:** data.yaml'ın, YOLO modelinin eğitim öncesinde ihtiyaç duyduğu temel referans dosyası olduğu; verilerin konumu (path, train, val, test), toplam sınıf sayısı (nc) ve sınıf isimlerinin (names) bu dosyadan okunduğu, veri yollarında veya sınıf bilgilerinde oluşacak küçük bir hatanın dahi modelin veri setini yanlış okumasına yol açabileceği tespit edilmiştir.

- **train, val ve test klasörlerinin amacı:** train (Eğitim) kümesinin modelin nesneleri tanımayı öğrendiği ve veri setinin en büyük kısmını (%80) oluşturan küme; val (Doğrulama) kümesinin her epoch sonunda modelin genelleme başarısını (overfitting olup olmadığını) izlemek için kullanılan küme; test kümesinin ise eğitim tamamen bittikten sonra modelin daha önce hiç görmediği verilerle nihai performansının ölçüldüğü küme olduğu bir öğrencinin sınava hazırlanma süreciyle benzetilerek özetlenmiştir.

- **Verilerin dağıtım mantığı:** Fotoğrafların train/val/test klasörlerine rastgele dağıtılması gerektiği; verinin kronolojik veya belirli bir örüntüye göre bölünmesi durumunda modelin yalnızca belirli koşullara özgü öğrenip genel performansta başarısız olabileceği, yaygın kabul gören oranın %80 Train, %10 Val, %10 Test olduğu belirlenmiştir.

- **labels klasöründeki .txt dosyalarının yapısı:** Bir etiket satırının (ör. 0 0.52 0.45 0.12 0.34) sırasıyla sınıf numarasını, kutunun merkez x-y koordinatlarını (görüntüye oranlı) ve kutunun genişlik-yükseklik değerlerini (görüntüye oranlı) ifade ettiği tespit edilmiştir.

- **images ve labels klasörleri arasındaki eşleşme zorunluluğu:** YOLO'nun veri okuma mekanizmasının doğrudan dosya yolu/isim eşleştirmesi üzerinden çalıştığı; images/train'deki bir görüntünün (ör. kare_15.jpg) yol ve uzantısının images→labels, .jpg→.txt şeklinde dönüştürülerek labels/train/kare_15.txt dosyasının arandığı; isim uyuşmazlığı veya yanlış klasörde bulunma durumunda modelin görüntüyü hatalı biçimde “nesne içermeyen arka plan” olarak yorumlayıp yanlış öğrendiği belirlenmiştir. Bu doğrultuda her görüntü-etiket çiftinin aynı isimle aynı (train/val/test) klasörde bulunmasına özenle uyulmuştur.

- **Veri seti hazırlama sürecinin önemi:** Model başarısının yalnızca kullanılan algoritmaya değil, etiketleme doğruluğu, görüntü-etiket eşleştirmesi, veri kümelerinin dengeli dağıtılması ve data.yaml'ın doğru yapılandırılmasına da bağlı olduğu; bu nedenle veri hazırlama sürecinin model geliştirmenin en önemli aşamalarından biri olduğu sonucuna varılmıştır.

**Gün 7 Özetle:** market_video.mp4'ten elde edilen kareler Label Studio ile tek sınıflı insan veri seti olarak etiketlenip YOLOv8 OBB formatında dışa aktarılmış, %80/%10/%10 oranında train/val/test klasörlerine organize edilmiştir. data.yaml'ın görevleri, veri dağılımının rastgele yapılmasının önemi, etiket dosyası yapısı ve görüntü-etiket eşleştirme mekanizması incelenmiş; Google Colab'da başlatılan eğitimde oluşan dosya yolu hatası analiz edilerek sonraki güne bırakılmıştır.

<span id="_Toc235230949" class="anchor"></span>Gün 8 --- 16.07.2026: Model Eğitiminin Tamamlanması, Metriklerin Yorumlanması ve Gerçek Zamanlı Kişi Sayımı

- **Yapılan İşlem:** Bir önceki gün hazırlanan veri seti kullanılarak YOLO modelinin eğitim süreci tamamlanmış; önce eğitim sırasında karşılaşılan dosya yolu (path) hatası giderilmiş, ardından model Google Colab'da başarıyla eğitilmiştir. Eğitim boyunca oluşan loss, precision, recall ve mAP metrikleri ile performans grafikleri (F1, PR eğrisi, Confusion Matrix, results.png, labels.jpg, train_batch görselleri) ayrıntılı olarak incelenmiş; epoch, learning rate, overfitting, underfitting, best fit, bias--variance dengesi ve early stopping gibi temel kavramlar uygulamalı değerlendirilmiştir. Günün sonunda eğitilen model kullanılarak gerçek zamanlı kişi sayımı (line crossing) uygulaması geliştirilmiştir.

- **Çalışmanın Amacı:** Eğitilen modelin performansının yalnızca tek bir sayısal değerle değil, tüm metrik ve grafiklerle birlikte yorumlanması ve modelin gerçek bir senaryoda (markette giriş-çıkış/kişi sayımı) test edilerek doğrulanması hedeflenmiştir.

- **Path (dosya yolu) hatasının giderilmesi:** Hatanın kök nedeninin göreceli yol (relative path) ile mutlak yol (absolute path) karışıklığı olduğu; data.yml içindeki path: ./ ifadesinin Colab'ın varsayılan çalışma dizini olan /content/'i işaret ettiği, oysa gerçek veri setinin Google Drive'da bulunduğu tespit edilmiştir. Çözüm olarak path değeri Drive üzerindeki veri setinin mutlak yoluyla (/content/drive/MyDrive/market_dataset) güncellenmiş; ayrıca dosya uzantısının .yaml değil .yml olduğunun (sistemin bu iki uzantıyı farklı dosyalar olarak değerlendirdiği) fark edilmesi gerektiği anlaşılmıştır. Drive üzerinden doğrudan okumada eğitimin ilk epoch'a kadar gecikebileceği, bunun hata değil normal bir Drive okuma performansı olduğu not edilmiştir.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p>from ultralytics import YOLO</p>
<p>model = YOLO('yolov8n.pt')</p>
<p>sonuclar = model.train(</p>
<p>data='/content/drive/MyDrive/market_dataset/data.yml',</p>
<p>epochs=50, imgsz=640, plots=True</p>
<p>)</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

- **Eğitim sonucu ve genel performans:** Model, yolov8n.pt (nano) mimarisiyle 50 epoch boyunca yaklaşık 4 dakika (0,068 saat) sürede eğitilmiştir. Elde edilen doğrulama sonuçları aşağıdaki gibidir; mAP50'nin 0,959 (%95,9) olması yüksek bir başarı seviyesine işaret etmekle birlikte, bu denli yüksek doğruluğun (genellikle %80-90 beklenirken %95'e ulaşılmasının) overfitting ihtimaline karşı dikkatle değerlendirilmesi gerektiği vurgulanmıştır.

| **Metrik**    | **Değer** |
|---------------|-----------|
| Precision (P) | 0,948     |
| Recall (R)    | 0,935     |
| mAP50         | 0,959     |
| mAP50-95      | 0,601     |

- **Epoch, learning rate ve overfitting/underfitting kavramları:** Epoch'un veri setinin tamamının bir kez işlenmesi, epochs=50'nin veri setinin 50 kez baştan sona okunacağı anlamına geldiği; öğrenme oranının (learning rate) modelin ağırlık güncelleme adım büyüklüğünü belirlediği, çok yüksek değerin optimum noktayı aşmaya (overshooting), çok düşük değerin ise eğitimin aşırı yavaşlamasına yol açtığı, modern eğitimde dinamik (scheduler) bir öğrenme oranının tercih edildiği öğrenilmiştir. Best Fit'te train ve val loss'un birlikte düşüp platoya oturduğu, overfitting'te train loss düşerken val loss'un bir noktadan sonra tekrar yükseldiği, underfitting'te ise her ikisinin de yüksek kaldığı; bu üç senaryoyu ayırt etmenin temel ölçütünün train-val loss arasındaki farkın (generalization gap) izlenmesi olduğu belirlenmiştir.

- **Bias/Variance dengesi ve temel performans metrikleri:** Yüksek bias'ın modelin çok basit kalıp gerçek ilişkiyi yakalayamamasına (underfitting), yüksek variance'ın ise modelin veriyi ezberleyip yeni veride kötü genelleme yapmasına (overfitting) karşılık geldiği incelenmiştir. Precision'ın (TP/(TP+FP)) modelin doğru dediği tespitlerin gerçekten doğru olma oranını, Recall'un (TP/(TP+FN)) gerçek nesnelerin ne kadarının yakalandığını, mAP50'nin IoU=0,5'te ortalama başarıyı, mAP50-95'in ise IoU 0,5-0,95 aralığındaki ortalamayı (modelin gerçek performansının en önemli göstergesi) ifade ettiği; box_loss, cls_loss ve dfl_loss değerlerinin sırasıyla kutu konumlandırma, sınıflandırma ve kutu kenarı hassasiyet hatasını gösterdiği ve eğitim boyunca azalmasının beklendiği belirlenmiştir.

- **Eğitim loglarının analizi:** 50 epoch boyunca box_loss'un 1,96'dan 1,10'a, cls_loss'un 3,07'den 0,60'a, dfl_loss'un 1,53'ten 1,02'ye düzenli biçimde azaldığı; precision'ın 0,03'ten 0,95'e, recall'un ~0,30'dan 0,93-0,94'e, mAP50'nin 0,34'ten 0,96'ya, mAP50-95'in 0,17-0,18'den 0,60'a yükselip son epoch'larda platoya oturduğu tespit edilmiştir. 40. epoch'ta Close-Mosaic aşamasına geçilerek mozaik veri artırmanın kapatıldığı ve albumentations üzerinden Blur, MedianBlur, ToGray, CLAHE gibi ek tekniklerin devreye girdiği görülmüştür. Train ve val loss eğrileri arasında zamanla açılan bir fark gözlemlenmediğinden modelin overfitting yapmadığı ve sağlıklı bir öğrenme (best fit) gerçekleştirdiği sonucuna varılmıştır.

- **Eğitim grafiklerinin yorumlanması:** F1-Confidence eğrisinde “all classes 0.94 at 0.503” değeriyle güven eşiği ~0,503'te maksimum %94 F1 skoruna ulaşıldığı; Precision-Confidence eğrisinde güven eşiği %92,7 ve üzerinde precision'ın %100'e ulaştığı; Precision-Recall eğrisinde “all classes 0.959 mAP@0.5” değeriyle recall %90'a kadar precision'ın %99 üzerinde kaldığı, %95 sonrası ani düşüş yaşandığı (zor/uzak nesneler nedeniyle); Recall-Confidence eğrisinde güven eşiği sıfıra indirildiğinde nesnelerin %98'inin yakalandığı gözlemlenmiştir. Confusion Matrix'te 193 örnekte insanların %96 oranında doğru tespit edildiği, arka planın %100 doğru filtrelendiği, 7 insanın kaçırıldığı (%3) ve 23 arka plan nesnesinin yanlışlıkla insan olarak işaretlendiği belirlenmiştir. labels.jpg'nin veri setinde 1700 “Human” etiketi bulunduğunu, kutuların çoğunlukla dikey (ayakta duran insan) formda ve orta/uzak mesafede yoğunlaştığını gösterdiği; results.png'de tüm loss ve metrik eğrilerinin overfitting bulgusu içermediği tespit edilmiştir.

- **train_batch görsellerinin incelenmesi:** Eğitimin ilk aşamalarında Mosaic Augmentation ile 4 farklı fotoğrafın tek karede birleştirildiği; mavi kutuların ve “0” (Human) etiketinin insanların üzerine kusursuz oturduğu; imgsz=640 sabitlemesi nedeniyle kenarlarda gri padding (letterboxing) oluştuğu; son 10 epoch'ta Close-Mosaic ile mozaiklemenin kapatılıp modelin orijinal fotoğraflarla ince ayar yaptığı; farklı kamera açılarında (yüksek tavan, fisheye) ve kalabalık/örtüşme durumlarında da tespitin başarılı olduğu gözlemlenmiştir. Kalabalık sahnelerde görsel kirliliği azaltmak için predict aşamasında show_labels=False ve show_conf=False kullanılabileceği not edilmiştir.

- **Overfitting'in önlenmesi ve early stopping:** Veri odaklı tekniklerden (daha fazla veri, Data Augmentation: Blur/CLAHE/Mosaic) ve model/algoritma odaklı tekniklerden (Early Stopping -- patience parametresi, Dropout, Regularization/Weight Decay, model kapasitesinin sınırlandırılması -- yolov8n tercihi) bahsedilmiştir. Overfitting'in canlı tespiti için train loss düşerken val loss'un yükselmeye başlamasının ve mAP50-95'in gerilemeye başlamasının kesin gösterge olduğu; mevcut eğitimde böyle bir bozulma gözlemlenmediğinden eğitimin sağlıklı tamamlandığı değerlendirilmiştir.

- **Uygulama -- markette anlık kişi sayısı takibi (line crossing):** Eğitilen model kullanılarak bir market videosunda kapıdan giren/çıkan kişilerin sayılması amacıyla supervision kütüphanesinin ByteTrack ve LineZone bileşenleri kullanılmıştır. Geliştirilen betik; yalnızca “person” sınıfını (classes=\[0\]) tespit etmekte, NMS ile süzmekte, ByteTrack ile takip etmekte ve kişilerin ayak noktalarının (BOTTOM_CENTER) tanımlı çizgiyi geçişini LineZone ile kontrol ederek anlık içeride bulunan kişi sayısını (negatife düşmeyecek şekilde) video üzerine yazdırmakta ve hem yerel ortamda hem Drive'da kaydetmektedir.

- **Model eğitim sürecinde edinilen deneyimler:** Model eğitiminin yalnızca model.train() komutunun çalıştırılmasından ibaret olmadığı; veri setinin doğru hazırlanması, data.yml'ın uygun yapılandırılması, öğrenme oranı/epoch sayısının doğru belirlenmesi, logların düzenli analiz edilmesi ve performans grafiklerinin doğru yorumlanmasının başarılı bir model geliştirme sürecinin temel bileşenleri olduğu; eğitim sonrası modelin gerçek uygulamalarda test edilmesinin de metriklerin doğrulanması açısından önemli olduğu görülmüştür.

**Gün 8 Özetle:** Veri seti kullanılarak YOLO modeli başarıyla eğitilmiş, precision/recall/mAP/loss metrikleri ve eğitim grafikleri (F1, PR eğrisi, Confusion Matrix, results.png, labels.jpg, train_batch) ayrıntılı yorumlanmış; epoch, learning rate, bias-variance, overfitting/underfitting, best fit ve early stopping kavramları uygulamalı incelenmiştir. Eğitilen model, ByteTrack ve LineZone ile geliştirilen gerçek zamanlı kişi sayımı uygulamasında başarıyla test edilmiştir.

<span id="_Toc235230950" class="anchor"></span>Gün 9 --- 17.07.2026: Klasik Görüntü Ön İşleme Teknikleri ve Haar Cascade ile Yüz Tespiti

- **Yapılan İşlem:** Derin öğrenme tabanlı nesne tespitine geçmeden önce uygulanan temel ön işleme (preprocessing) teknikleri incelenmiştir: konvolüsyon tabanlı filtreleme/bulanıklaştırma yöntemleri, kenar tespiti algoritmaları (Sobel, Canny), otomatik eşikleme (Otsu Thresholding), kontur tespiti, histogram tabanlı eşleştirme yöntemleri ve klasik makine öğrenmesi tabanlı yüz tespiti (Haar Cascade) yaklaşımı. Öğrenilen bilgilerin pekiştirilmesi amacıyla OpenCV ile HSV renk uzayında kırmızı bir topun tespit ve takip uygulaması da gerçekleştirilmiştir.

- **Çalışmanın Amacı:** Klasik görüntü işleme algoritmalarının çalışma prensiplerinin öğrenilmesi, görüntü ön işleme adımlarının önemini kavranması ve OpenCV'nin temel fonksiyonlarının kullanım mantığının, teorik bilgiyi uygulamayla destekleyerek anlaşılması hedeflenmiştir.

- **Konvolüsyon ve kernel kavramı:** Filtreleme işlemlerinin temelinde, görüntü üzerinde küçük bir pencerenin (kernel/maske, genellikle 3x3 veya 5x5) gezdirilerek her konumdaki merkez pikselin komşu piksellerin ağırlıklı ortalamasıyla yeniden hesaplanması (konvolüsyon) yer aldığı öğrenilmiştir. Correlation'da kernelin olduğu gibi, convolution'da ise kernelin 180 derece döndürülerek uygulandığı; kernel içindeki sayısal değerlerin görüntünün bulanıklaştırılıp bulanıklaştırılmayacağını, kenarların mı bulunacağını yoksa keskinleştirilip keskinleştirilmeyeceğini belirlediği (Edge Detection, Sharpen, Blur, Identity, Highpass, Emboss, Sobel kernelleri) tespit edilmiştir.

- **Temel bulanıklaştırma algoritmaları:** Box Filter'ın tüm piksellere eşit ağırlık vererek kaba/yapay bir bulanıklık ürettiği (cv2.blur); Gaussian Blur'ün merkeze yüksek, uzaklaştıkça azalan ağırlık vererek daha doğal bir bulanıklık sağladığı ve kenar tespiti öncesi mikro detay bastırmada en sık tercih edildiği (cv2.GaussianBlur); Median Blur'ün piksel değerlerinin medyanını seçerek özellikle tuz-biber gürültüsünün giderilmesinde etkili olduğu (cv2.medianBlur); Bilateral Filter'ın hem mekânsal hem yoğunluk farkına duyarlı iki Gaussian dağılımını birlikte kullanarak kenarları koruyup iç yüzeyleri pürüzsüzleştirdiği, ancak diğer yöntemlere göre belirgin şekilde daha yavaş çalıştığı incelenmiştir.

| **Filtre/Blur** | **Mantığı**                                 | **En Uygun Kullanım Alanı**                                 |
|-----------------|---------------------------------------------|-------------------------------------------------------------|
| Box Filter      | Tüm piksellere eşit ağırlık (düz ortalama)  | Hızlı ve kaba hesaplamalar                                  |
| Gaussian        | Merkeze yüksek ağırlık (ağırlıklı ortalama) | Nesne tespiti öncesi mikro detay/doku bastırma              |
| Median          | Ortanca (medyan) değeri seçme               | Sensör kaynaklı parazit/ölü piksel temizliği                |
| Bilateral       | Renk farkına duyarlı, kenar koruma          | Segmentasyon öncesi kenar bozulmadan yüzey pürüzsüzleştirme |

- **Kenar tespiti -- Sobel ve Canny algoritmaları:** Kenarın, görüntü fonksiyonundaki ani değişimleri (iki farklı parlaklık/renk bölgesi arasındaki sınırı) ifade ettiği; kenar tespiti öncesinde gürültünün türev hesaplamasını bozmaması için mutlaka smoothing uygulanması gerektiği belirlenmiştir. Sobel algoritmasının Gx (yatay) ve Gy (dikey) olmak üzere iki 3x3 konvolüsyon matrisiyle türevi hesaplayıp Pisagor teoremiyle kenarın şiddet ve yönünü bulduğu; Canny algoritmasının ise Sobel'in kalın/gürültülü çıktısını (1) Gaussian Blur ile gürültü azaltma, (2) Sobel ile gradyan hesaplama, (3) Non-Maximum Suppression ile inceltme, (4) çift eşikleme (hysteresis thresholding) aşamalarından geçirerek ince ve net sınırlara dönüştürdüğü (cv2.Canny) öğrenilmiştir.

- **Otsu eşikleme (Otsu Thresholding):** Sabit eşiklemenin ışık koşulları değiştiğinde başarısız olduğu; Otsu algoritmasının görüntünün histogramına bakarak sınıf-içi varyansı en aza indiren eşik değerini otomatik bulduğu, histogramın iki belirgin tepe (bimodal) gösterdiği durumlarda (belge tarama/OCR, tıbbi görüntüleme, üretim bantları) etkili olduğu; tekdüze olmayan aydınlatmada (yalnızca bir bölgeye düşen gölge) ise Otsu'nun da başarısız olup Adaptive Thresholding gerektiği tespit edilmiştir. Önerilen işlem sırasının gri tonlama + hafif Gaussian Blur → segmentasyon (Canny/Thresholding) → kontur tespiti (vektörleştirme) → analiz/çıkarım şeklinde ilerlediği belirlenmiştir.

- **Kontur tespiti (Contour Detection):** Kenar/eşikleme ile ayrıştırılan nesnelerin cv2.findContours ile geometrik koordinat listelerine dönüştürüldüğü; hiyerarşi modlarından RETR_EXTERNAL'in yalnızca en dış konturları, RETR_TREE'nin tüm konturları hiyerarşik olarak aldığı; yaklaşım yöntemlerinden CHAIN_APPROX_SIMPLE'ın düz çizgilerin yalnızca uç noktalarını tutarak bellek kullanımını azalttığı öğrenilmiştir. cv2.approxPolyDP (Douglas-Peucker) ile köşe sayısına göre şekil tanıma (3 köşe: üçgen, 4 köşe: dörtgen, 8+ köşe: daire/elips) yapılabildiği; cv2.contourArea ve cv2.arcLength ile alan/çevre ölçümü yapılabildiği, piksel değerlerinin gerçek ölçüye (cm²) çevrilmesi için bilinen boyutlu bir referans nesneye ihtiyaç duyulduğu ve tek bir 2D konturdan gerçek 3D hacmin doğrudan hesaplanamayacağı (derinlik bilgisi eksikliği) belirlenmiştir.

- **Histogramlar ve görüntü eşleştirme:** Renk histogramlarının bir görüntüde hangi renkten kaç piksel olduğunu gösterdiği, ışık değişimlerine karşı analizin genellikle HSV uzayında yapıldığı; mekânsal bilgiyi kaybetmesinin nesnenin dönme/kısmi kapanma durumlarında avantaja dönüştüğü (CamShift, MeanShift ile takipte kullanıldığı) öğrenilmiştir. Histogram eşlemenin (matching) bir kaynak görüntünün renk/ışık karakteristiğini CDF üzerinden bir referans görüntüye aktardığı (stereo kamera kalibrasyonu, gündüz→gece veri üretimi); Template Matching'in (cv2.matchTemplate) küçük bir şablonu kaydırarak (sliding window) en yüksek benzerlik skorunu bulduğu, ancak ölçek/rotasyon değişiminde başarısız olduğu için günümüzde SIFT/SURF/ORB veya derin öğrenme modellerinin tercih edildiği belirlenmiştir.

- **Yüz tespiti ve Haar Cascade (Viola-Jones) mimarisi:** Yüz tespitinin (kimin olduğunu değil) yalnızca yüzün varlık/konumunu belirlediği; genel işleyişin ön işleme → özellik çıkarımı → sınıflandırma → konumlandırma → son işleme adımlarından oluştuğu incelenmiştir. 2001'de Viola ve Jones tarafından geliştirilen Haar Cascade'in dört temel bileşeni: (a) Haar Özellikleri -- aydınlık/karanlık bölge oranını ölçen siyah-beyaz şablonlar (ör. göz bölgesinin yanaktan koyu olması); (b) İntegral Görüntü -- herhangi bir dikdörtgen bölgenin piksel toplamının sabit sürede (O(1)) hesaplanmasını sağlayarak gerçek zamanlı çalışmayı mümkün kılan matris; (c) AdaBoost -- 160.000'den fazla olası özellik arasından yüzü en iyi tanımlayan sınırlı sayıdakini (~6.000) seçen algoritma; (d) Cascade (Şelale) yapısı -- ilk aşamalarda az özellikle yüz olmayan pencereleri hızla eleyip yalnızca yüze benzeyen bölgelere işlem gücü ayıran kademeli mimari olarak tespit edilmiştir. OpenCV'de detectMultiScale() fonksiyonunun (x, y, w, h) koordinatlarını döndürdüğü belirlenmiştir.

- **Haar Cascade'in gözlemlenen sınırlamaları:** Baş yana yatırıldığında (~15-20 dereceden fazla eğimde) yatay/dikey eksenli Haar şablonlarının göz bölgesine oturmaması nedeniyle tespitin başarısız olduğu (eğim/rotasyon sorunu); haarcascade_frontalface_default.xml'in yalnızca tam karşıdan bakan yüzler için eğitildiği, baş çevrildiğinde simetrinin bozulmasıyla tespitin başarısız olduğu (profil sorunu, ayrı bir haarcascade_profileface.xml gerektiği); şelale yapısında yüzün bir kısmı kapatıldığında (elle vb.) ilgili aşamada işlemin anında reddedildiği (occlusion sorunu) gözlemlenmiştir. Bu sınırlamaların, endüstrinin Haar Cascade'den CNN/YOLO gibi derin öğrenme tabanlı modellere geçişinin somut nedenlerini oluşturduğu değerlendirilmiştir.

| **Özellik**         | **Haar Cascade**                      | **YOLO (ve diğer CNN'ler)**          |
|---------------------|---------------------------------------|--------------------------------------|
| Teknoloji           | Klasik makine öğrenmesi (AdaBoost)    | Derin öğrenme (CNN)                  |
| Çalışma Mantığı     | Sliding window (piksel piksel tarama) | Grid regression (tek seferde işleme) |
| Donanım İhtiyacı    | Yalnızca CPU'da dahi hızlı çalışır    | Verimli çalışma için GPU gereklidir  |
| Açı/Rotasyon        | Zayıf; yalnızca dümdüz bakan yüzler   | Güçlü; yan/kısmen kapalı/eğik yüzler |
| Işık Hassasiyeti    | Yüksek hassasiyet                     | Farklı aydınlatmayı iyi tolere eder  |
| Yanlış Tespit Oranı | Yüksek                                | Düşük                                |

- **Haar Cascade ile gerçek zamanlı yüz tespiti uygulaması:** OpenCV'nin hazır haarcascade_frontalface_default.xml modeliyle kameradan alınan görüntü gri tonlamaya çevrilip detectMultiScale() ile taranmıştır (scaleFactor: pencere büyütme oranı, minNeighbors: doğrulama sayısı, minSize: en küçük yüz boyutu); tespit edilen her yüzün (x, y, w, h) koordinatlarıyla orijinal görüntü üzerine yeşil bir dikdörtgen çizildiği, algoritmanın donanımı zayıf ve kameraya dümdüz bakan sabit koşullu senaryolarda; YOLO/modern CNN'lerin ise hareketli kamera, farklı açı, değişken ışık ve yüksek doğruluk gereken senaryolarda tercih edilmesi gerektiği sonucuna varılmıştır.

- **Uygulama -- renk tabanlı nesne (kırmızı top) takibi:** HSV renk uzayında kırmızı bir topun tespiti için önce H/S/V aralıkları en geniş halinde başlatılıp Hue kademeli daraltılarak (kırmızı ~0-15 / 160-179), ardından Saturation ve Value alt sınırları (ör. S: 110→50, V: 95→50) esnetilerek kalibre edilmiş; nihai aralık lower_red=\[0,50,50\], upper_red=\[15,255,255\] olarak belirlenmiştir. Uygulama akışında her karede Gaussian Blur ile gürültü azaltma, BGR→HSV dönüşümü, cv2.inRange ile maskeleme, erozyon/dilatasyon ile maske temizleme, cv2.findContours ile dış kontur tespiti, en büyük konturun top kabul edilmesi, cv2.minEnclosingCircle ve cv2.moments ile merkez/yarıçap hesaplanması ve sonucun bir video dosyasına kaydedilmesi adımları uygulanmıştır.

**Gün 9 Özetle:** Konvolüsyon ve kernel kavramlarıyla başlayan klasik görüntü işleme hattı; filtreleme (Box, Gaussian, Median, Bilateral), kenar tespiti (Sobel, Canny), Otsu eşikleme, kontur tespiti, histogram/şablon eşleştirme ve Haar Cascade tabanlı yüz tespiti bütünsel olarak ele alınmış, Haar Cascade'in YOLO'ya kıyasla güçlü/zayıf yönleri karşılaştırılmıştır. Elde edilen teorik bilgi, HSV tabanlı kırmızı top takibi uygulamasıyla pekiştirilmiştir.

# 3. Karşılaşılan Zorluklar / Öğrenilenler

# 

Bu hafta karşılaşılan teknik zorluklar ve bunlardan çıkarılan dersler özetlenmektedir (detaylı anlatımları ilgili gün başlıkları altında yer almaktadır):

| **Yaşanan Zorluk**                                                                                                                                                                                                     | **Çözüm / Öğrenilen Ders**                                                                                                                                                                                                                  |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Kalabalık ve hareketli kamera sahnelerinde standart takip algoritmalarının (ör. ByteTrack) yalnızca konum bilgisine dayanması nedeniyle kimlik karışması (ID swapping) riski taşıması.                                 | ReID desteğine sahip Deep OC-SORT ve TrackTrack algoritmalarının bu tür sahneler için daha uygun olduğu belirlenmiş; with_reid: True ayarı ve appearance_thresh yükseltilerek kimlik sürekliliği artırılmıştır.                             |
| SAHI algoritmasında yüksek doğruluk sağlayan yolov8x modelinin, her kareyi çok sayıda dilime bölerek ayrı çıkarım yapması nedeniyle işlem süresini ciddi ölçüde artırması.                                             | slice_height/slice_width değerleri 512-640 aralığında, overlap oranları 0,2-0,3 aralığında tutulmuş; GPU olmayan ortamlarda video kare kare işlenip ayrı dosya olarak kaydedilmesi yaklaşımı benimsenmiştir.                                |
| Model eğitimi sırasında data.yml içindeki göreceli yol (path: ./) ile Google Drive'daki gerçek veri konumunun uyuşmaması nedeniyle dosya yolu (path) hatası alınması ve eğitimin tamamlanamaması.                      | path değeri Drive üzerindeki verinin mutlak yoluyla (/content/drive/MyDrive/market_dataset) güncellenmiş; ayrıca dosya uzantısının .yaml değil .yml olduğuna dikkat edilmesi gerektiği fark edilmiştir.                                     |
| Markette kişi sayımı uygulamasında (LineZone) takip kimliğinin (ID) sürekli değişmesi nedeniyle sistemin aynı kişiyi her seferinde “yeni kişi” sayması ve içeride bulunan kişi sayısının hiç değişmemesi.              | Güven eşiği (conf=0,35-0,45 aralığı) optimize edilmiş; ByteTrack'in lost_track_buffer değeri (90-120 kareye kadar) artırılarak kısa süreli görünmezliklerde ID kaybı önlenmiş, debug amacıyla ID ve sayaç değerleri konsola yazdırılmıştır. |
| Sayım çizgisinin koordinatlarının, videonun gerçek çözünürlüğünden farklı bir ekran (ör. Retina/4K) üzerinden ölçülmesi nedeniyle çizginin işlenen görüntü alanının tamamen dışında kalması (“görünmez çizgi” hatası). | Çizgi koordinatları, işlenen videonun gerçek çözünürlüğüne (ör. 1280x720 / 1920x1080) göre yeniden hesaplanarak tanımlanmıştır.                                                                                                             |
