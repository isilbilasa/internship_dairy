import cv2 as cv 
import numpy as np

def add_info_bar(image, text, position="top"):
    h, w = image.shape[:2] # resmin yüksekliğini ve genişliğini alıyoruz
    bar_height = 40        # çizilecek siyah kutunun yüksekliği
    
    if position == "top":
        # overlay
        cv.rectangle(image, (0, 0), (w, bar_height), (0, 0, 0), -1)
        cv.putText(image, text, (15, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
        
    elif position == "bottom":
        # status bar
        cv.rectangle(image, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        cv.putText(image, text, (15, h - 12), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)


def mouse(event, x, y, flags, param):
    text =f'mouse at ({x}, {y}) flags = {flags}, param = {param}'
    add_info_bar(img,text,position="bottom")
    cv.imshow('window', img)

def rgb(x):
    r = max(0 , cv.getTrackbarPos('red','window'))
    g = max(0, cv.getTrackbarPos('green','window'))
    b = max(0, cv.getTrackbarPos('blue','window'))

    img[:] = [b,g,r]
    add_info_bar(img,f'Red={r}, Green={g}, Blue={b}',position= "top")
    cv.imshow('window',img)
    
img = np.zeros((100,600,3), 'uint8')
cv.imshow('window',img)

cv.setMouseCallback('window', mouse)

cv.createTrackbar('red', 'window', 200,255,rgb) # 0..255 , başlangıç 100
cv.createTrackbar('green', 'window', 50,255,rgb)
cv.createTrackbar('blue', 'window', 100,255,rgb)
rgb(0)

cv.waitKey(0)
cv.destroyAllWindows()