import cv2 
import mediapipe as mp 
import time 
import speech_recognition as sr 

mp_drawing = mp.solutions.drawing_utils # El izleme işlevlerini ve çizim yardımcılarını içe aktardık.
mp_drawing_styles = mp.solutions.drawing_styles # Çizim stillerini içe aktardık, 
mp_hands = mp.solutions.hands # Kütüphaneden el izleme modelini içe aktarır.

# Web kamerası girişi için:
cap = cv2.VideoCapture(0)
hand_closed = False  # Elin kapalı olup olmadığını belirten bayrak

# Önceki parmak sayısını ve sayaç için zaman değişkenlerini tanımlayın
previous_finger_count = 0
counter_start_time = time.time() # Kodun iyileştirmeye açık olması için zaman modulünü başlattık
recorded_count = None # Parmak Sayısını kaydettik.
geri_sayim = None # Fotoğraf süresini belirlemek için değişken oluşturduk.
flash_opened = None # Flaş geri bildirimi için değişken tanımladık.
exit_program = None

r = sr.Recognizer() # Ses tanıma işlemleri için bir Recognizer nesnesi oluşturduk.
mic = sr.Microphone() # Bir Microphone nesnesi oluşturduk, mikrofon erişimi sağlar.

with mic as source: # Mikrofon erişimi için değişkeni kaynak olarak kullandık.
        print("Sesinizi bekliyorum...")
        r.adjust_for_ambient_noise(source)  # Gürültüyü azalt

        audio = r.listen(source)  # Ses girişini dinle
        try:
            text = r.recognize_google(audio, language="tr-TR")  # Ses girişini metne dönüştür
            print(f"Ses algilandi: {text}")

            # Kullanıcı komutlarına göre kontrol yap
            if "flaş aç" in text.lower():
                print("Flaş açildi!")
                flash_opened = True
                exit_program = False
                # Buraya flaşı açma kodu eklenebilir
            elif "flaş kapat" in text.lower():
                print("Flaş kapatildi!")
                flash_opened = False
                exit_program=False
                # Buraya flaşı kapama kodu eklenebilir
            else: 
               print("TANIMLANMAYAN KOMUT!")
               exit_program = True   

        except sr.UnknownValueError:
            print("TANIMLANMAYAN KOMUT!")
            exit_program = True
        except sr.RequestError:
            print("Ses servisi çalişmiyor !")
            exit_program = True

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened() and (geri_sayim is None or geri_sayim >0) and exit_program == False:
    success, image = cap.read()
    if not success:
      print("Boş kamera görüntüsünü görmezden geliyor.")
      # Bir video yüklüyorsanız, 'continue' yerine 'break' kullanın.
      continue

    # Performansı artırmak için, görüntüyü referans iletmek üzere yazılamaz olarak işaretleyin.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Görüntü üzerine el işaretlerini çizin.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Başlangıçta her bir şapka için parmak sayısını 0 olarak ayarlayın
    fingerCount = 0

    if results.multi_hand_landmarks:

      for hand_landmarks in results.multi_hand_landmarks:
        # Etiketi kontrol etmek için el indeksini alın (sol veya sağ)
        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
        handLabel = results.multi_handedness[handIndex].classification[0].label

        # Landmark pozisyonlarını (x ve y) tutmak için değişken ayarlayın
        handLandmarks = []

        # Her landmark'ın x ve y pozisyonlarını listeleyin
        for landmarks in hand_landmarks.landmark:
          handLandmarks.append([landmarks.x, landmarks.y])

        # Her parmak için test koşulları: Parmağın kaldırıldığı düşünülüyorsa sayı arttırılır.
        # Başparmak: TIP x pozisyonu IP x pozisyonundan büyük veya küçük olmalı, el etiketine göre.
        if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
          fingerCount = fingerCount+1
        elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
          fingerCount = fingerCount+1

        # Diğer parmaklar: TIP y pozisyonu, resmin kökeninin sol üst köşesinde olduğundan daha küçük olmalıdır.
        if handLandmarks[8][1] < handLandmarks[6][1]:       # İşaret parmağı
          fingerCount = fingerCount+1
        if handLandmarks[12][1] < handLandmarks[10][1]:     # Orta parmak
          fingerCount = fingerCount+1
        if handLandmarks[16][1] < handLandmarks[14][1]:     # Yüzük parmağı
          fingerCount = fingerCount+1
        if handLandmarks[20][1] < handLandmarks[18][1]:     # Serçe parmağı
          fingerCount = fingerCount+1

        # El işaretlerini çizin
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Elin kapalı olup olmadığını kontrol et
      if fingerCount == 0: 
        hand_closed = True

    # Elin kapalı olduğunda kamerayı kapat
    if hand_closed:
        cv2.putText(image, "El kapali, kamera kapatiliyor...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('MediaPipe Hands', image)
        cv2.waitKey(2000)  # 2 saniye beklemek için
        break

    # Eğer parmak sayısı değişmişse, sayaç ve önceki sayıyı güncelle
    if fingerCount != previous_finger_count:
        counter_start_time = time.time()
        previous_finger_count = fingerCount

    # Eğer parmak sayısı 6 saniye boyunca değişmezse, kaydedin
    if fingerCount > 0:
       if time.time() - counter_start_time >=2 and fingerCount == previous_finger_count:
          recorded_count = fingerCount


    # Eğer kayıtlı sayı varsa, kontrol et
    if recorded_count is not None and recorded_count >= 0 :
        # Eğer kayıtlı sayı 3 saniye boyunca değişmezse, geri_sayim değişkenine ata
        if geri_sayim is None and time.time() - counter_start_time >= 3:
            geri_sayim = recorded_count

    # Eğer geri_sayim değişkeni atanmışsa, ekrana yazdır
    if geri_sayim is not None:
        cv2.putText(image, f"Geri Sayim: {geri_sayim}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Eğer geri sayım başladıysa
        if geri_sayim > 0:
            cv2.putText(image, f"Geri Sayim: {geri_sayim}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Her bir saniyede geriye doğru say
            if time.time() - counter_start_time >= 1:
                geri_sayim -= 1
                counter_start_time = time.time()  # Süreyi tekrar başlat

            # Geri sayım bittiğinde fotoğrafı çek
            if geri_sayim == 0:
                success, frame = cap.read()
                if success:
                    if flash_opened == True:
                       cv2.imwrite("flashacikfoto.jpg", frame)  # Çekilen fotoğrafı "foto.jpg" olarak kaydet
                       print("Fotoğraf flaş açıkken çekildi!")
                    else:
                       flash_opened == False
                       cv2.imwrite("flashkapalifoto.jpg", frame)  # Çekilen fotoğrafı "foto.jpg" olarak kaydet
                       print("Fotoğraf flaş kapalıyken çekildi!")
                    break

    

    # El durumunu (kapalı) ve parmak sayısını görüntüleyin
    cv2.putText(image, f"Parmak Sayisi: {fingerCount}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow('MediaPipe Hands', image)

    # Video penceresini kapatmak için
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()