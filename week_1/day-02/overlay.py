import cv2 as cv

file = 'images-3.jpeg'

#cv.IMREAD_COLOR, resmi renkli okumayı söyleyen bir sabittir. 
# (cv.IMREAD_GRAYSCALE ise gri okur.)
img = cv.imread(file, cv.IMREAD_COLOR)

text = f'file name: {file}'
font = cv.FONT_HERSHEY_SIMPLEX # yazı tipi
org = (15, 35)                 # yazının başlayacağı X ve Y koordinatı
fontScale = 1                  # boyutu
color = (0, 255, 0)            # rengi bgr formatında 
thickness = 1                 # kalınlığı

cv.putText(img,text,org,font,fontScale,color,thickness,cv.LINE_AA)
cv.imshow('window', img)

cv.waitKey(0)
cv.destroyAllWindows()