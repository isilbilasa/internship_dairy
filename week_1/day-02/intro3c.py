import cv2 as cv

img = cv.imread('images-3.jpeg')
# y=250..300, x=50..550 arası yeşil yap
img[250:300, 50:550] = (0,255,0) #yatay yeşil bant
cuted_img = img[80:190, 270:320]  
img[0:150, 0:120] = cuted_img

#slice [y_başı:y_sonu, x_başı:x_sonu] sırasındadır 