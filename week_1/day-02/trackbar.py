import cv2 as cv 

def trackbar(x):      #callback: x = sürgünün yeni değeri
    text = f'Trackbar: {x}'
    fontScale = 1

    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img,text,font,fontScale,cv.LINE_AA)
    cv.imshow('window',img)

img = cv.imread('images-3.jpeg',cv.IMREAD_COLOR)
cv.imshow('window',img)
cv.createTrackbar('x', 'window', 100,255,trackbar) # 0..255 , başlangıç 100

cv.waitKey(0)
cv.destroyAllWindows()