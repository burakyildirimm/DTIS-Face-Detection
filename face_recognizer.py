import cv2


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    coords = []
    
    # Resmi gri forma çevirdik
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Yüz üzerindeki nesneleri (xml dosyalarımızdan) tanıyıp bunların scala özelliklerini veriyor
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    
    # Burada yüzü tanımlayıcı kare çizdik ve kişinin ismini label ile gösterdik
    # Eğer şüpheli ise listede ise kırmızı değil ise yeşil yanacak
    
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        
        cv2.putText(img, "Burak", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]

    return coords

# Belirlenen suratın çizilmesini istenen özellikler ile tetikleyen fonksiyon yazıldı
def recognize(img, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color["red"], "Face")
    return img


# Yüzün tanımlanması burada oluyor
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 0 ile dahili kameraya ulaşıp video kaydını başlatıyoruz ! (1 ile harici kamera)
video_capture = cv2.VideoCapture(0)

# Videodan sonsuz bir ekran görüntüsü döngüsü başlattık !
while True:
    # Kameradan gelen kareleri fotoğraf olarak okuma !
    _, img = video_capture.read()
    img = recognize(img,faceCascade)
    # Yüz tanıma penceremiz tetiklendi !
    cv2.imshow("face detection", img)
    
    # Çıkış için q tuşu kullanılabilir !
    if cv2.waitKey(1) == ord('q'):
        break

# Video kaydı burada başladı !
video_capture.release()
cv2.destroyAllWindows()
