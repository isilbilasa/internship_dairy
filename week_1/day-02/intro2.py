import cv2 as cv 

cap = cv.VideoCapture(0)  #0 = varsayılan kamera 

while True: 
    ret,frame = cap.read()   #kameradan bir kare oku. cap.read() iki şey döndürür: ret (kare başarıyla okundu mu? True/False) ve frame (görüntünün kendisi).
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #kareyi griye çevir

    cv.imshow('frame', frame) #kareyi göster 
    if cv.waitKey(1) & 0xFF == ord('q'): #'q' tuşuna basılırsa 
        break     #döngüden çık

cap.release()    #kamerayı serbest bırak
cv.destroyAllWindows()