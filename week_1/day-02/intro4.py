import cv2 as cv

class App:                          #kurucu: nesne oluşunca çalışır 
    def __init__(self):
        img = cv.imread('images-3.jpeg')
        Window('image', img)
    
    def run(self):                  # ana olay döngüsü
        k = 0
        while k != ord('q'):
            k = cv.waitKey(0)
            print(k,chr(k))        # tuş kodu ve karakterini yaz 
        cv.destroyAllWindows

class Window:
    """Bir resimle pencere oluştur."""
    def __init__(self, win, img):
        self.win = win
        self.img = img
        cv.imshow(win,img)

if __name__ == '__main__':
    App().run()                 # appi başlat ve çalıştır 