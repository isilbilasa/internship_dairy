import cv2 as cv 

img = cv.imread('images-3.jpeg')
cv.imshow("window",img)

cv.waitKey(0)
cv.destroyAllWindows()