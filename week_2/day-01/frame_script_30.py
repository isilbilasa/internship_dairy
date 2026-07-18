import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=30):
    """
    Video dosyasını belirtilen aralıklarla frame'lere ayırır.
    :param video_path: Videonun dosya yolu
    :param output_folder: Çıktı klasörü
    :param frame_interval: Kaç karede bir kaydedileceği (default 30)
    """
    
    # çıktı klasörünü oluşturdum
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Klasör oluşturuldu: {output_folder}")

    # videoyu açtım
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video dosyası açılamadı!")
        return

    frame_count = 0
    saved_count = 0

    print("İşlem başlıyor...")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        # belirtilen aralıkta ise kaydettik
        if frame_count % frame_interval == 0:
            output_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(output_filename, frame)
            saved_count += 1
        
        frame_count += 1

    cap.release()
    print(f"İşlem tamamlandı. Toplam {saved_count} kare '{output_folder}' klasörüne kaydedildi.")

# Kullanım
video_dosyasi = "people_in_üsküdar.mp4"
cikti_klasoru = "dataset_frames4"
# her 30 karede bir kaydet (Saniyede 30 fps video için saniyede 1 kare alır)
extract_frames(video_dosyasi, cikti_klasoru, frame_interval=30)