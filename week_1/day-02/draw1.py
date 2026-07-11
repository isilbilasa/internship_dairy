import cv2 as cv 
import numpy as np

#sıfırdan boş bir resmi numpy ile üretiyoruz
img = np.zeros((100,500,3), np.uint8)  #renkli(3 kanal) siyah resim  
cv.imshow('RGB', img)

gray_img = np.zeros((100,500), np.uint8)  #gri(kanalsız) siyah resim
cv.imshow('Gray', gray_img)

cv.waitKey(0)
cv.destroyAllWindows()