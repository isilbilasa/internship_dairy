import cv2
import os

def extract_all_frames(video_path, output_folder):
    # çıktı klasörünü oluştur
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # videoyu aç
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video dosyası açılamadı!")
        return

    frame_count = 0

    print("İşlem başlıyor: Tüm kareler ayıklanıyor...")

    while True:
        ret, frame = cap.read()
        
        # video bittiğinde döngüden çık
        if not ret:
            break

        # her kareyi kaydet
        output_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_filename, frame)
        
        frame_count += 1

    cap.release()
    print(f"İşlem tamamlandı. Toplam {frame_count} kare '{output_folder}' klasörüne kaydedildi.")


video_dosyasi = "people_in_üsküdar.mp4"  # dosya adını buraya yaz
cikti_klasoru = "dataset_frames"

extract_all_frames(video_dosyasi, cikti_klasoru)