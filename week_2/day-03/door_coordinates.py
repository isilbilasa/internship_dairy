import cv2 as cv 

image_path = "market_kare.png"
img = cv.imread(image_path)

def coordinate_select(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f"Tıklanan koordinat -> X: {x}, Y: {y}")

        cv.circle(img, (x,y), 5, (0,0,255), -1)
        cv.imshow("koordinat bulucu",img)

cv.namedWindow("koordinat bulucu")
cv.setMouseCallback("koordinat bulucu", coordinate_select)

print("Kapının sol sınırına ve sağ sınırına tıklayarak terminaldeki koordinatları not al.")
print("Çıkmak için klavyeden 'q' veya 'ESC' tuşuna bas.")

while True:
    cv.imshow("koordinat bulucu", img)
    key = cv.waitKey(1) & 0x77
    if key == ord("q") or key == 27:
        break

cv.destroyAllWindows()