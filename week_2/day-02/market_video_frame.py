import cv2
import os

def extract_frames_hardened(video_path, output_folder, frame_interval=9):
    # Klasörün mutlak yolunu (tam adresini) al
    abs_output_folder = os.path.abspath(output_folder)
    
    # Klasör yoksa oluştur
    if not os.path.exists(abs_output_folder):
        os.makedirs(abs_output_folder)
        print(f"Klasör oluşturuldu: {abs_output_folder}")
    else:
        print(f"Klasör zaten mevcut: {abs_output_folder}")

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"HATA: Video dosyası açılamadı! Lütfen {video_path} dosyasının yerini kontrol edin.")
        return

    frame_count = 0
    saved_count = 0

    print("İşlem başlıyor...")

    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        if frame_count % frame_interval == 0:
            # Tam dosya yolunu oluştur
            file_name = f"frame_{saved_count:04d}.jpg"
            full_path = os.path.join(abs_output_folder, file_name)
            
            # Görüntüyü kaydet ve başarılı mı diye kontrol et
            success = cv2.imwrite(full_path, frame)
            
            if success:
                saved_count += 1
            else:
                print(f"hata: Kaydedilemedi -> {full_path}")
        
        frame_count += 1

    cap.release()
    print("-" * 30)
    print(f"İşlem tamamlandı.")
    print(f"Toplam kaydedilen kare: {saved_count}")
    print(f"Klasörün tam yolu: {abs_output_folder}")


extract_frames_hardened("market_video.mp4", "etiketlenecek_market_kareleri2", frame_interval=9)