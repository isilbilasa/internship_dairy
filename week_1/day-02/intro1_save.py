import cv2 as cv 

img = cv.imread("images-3.jpeg")
cv.imshow("window", img)

cv.imwrite("images-3.png",img) #png olarak kaydet
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) #gri tonlamaya çevirdim
cv.imwrite("images-gray.png",gray) #griyi kaydet.cv.imwrite('messi_gray.png', gray)

cv.waitKey(0)
cv.destroyAllWindows()