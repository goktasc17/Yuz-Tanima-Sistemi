import cv2
import face_recognition

foto1path = input("Fotografiniz cihazinizda nerede bulunuyor: ")
foto2path = input("Karsilastirmak istediginiz fotograf cihazinizda nerede bulunuyor: ")

fotograf1 = face_recognition.load_image_file(foto1path)
fotograf1 = cv2.cvtColor(fotograf1, cv2.COLOR_BGR2RGB)
fotograf2 = face_recognition.load_image_file(foto2path)
fotograf2 = cv2.cvtColor(fotograf2, cv2.COLOR_BGR2RGB)

YuzLokasyonu = face_recognition.face_locations(fotograf1)[0]
encodeImg1 = face_recognition.face_encodings(fotograf1)[0]
cv2.rectangle(fotograf1, (YuzLokasyonu[3], YuzLokasyonu[0]), (YuzLokasyonu[1], YuzLokasyonu[2]), (255, 0, 255), 2)

yuzLokasyonTest = face_recognition.face_locations(fotograf2)[0]
encodeImg2 = face_recognition.face_encodings(fotograf2)[0]
cv2.rectangle(fotograf2, (yuzLokasyonTest[3], yuzLokasyonTest[0]), (yuzLokasyonTest[1], yuzLokasyonTest[2]), (255, 0, 255), 2)

sonuclar = face_recognition.compare_faces([encodeImg1], encodeImg2)
faceDis = face_recognition.face_distance([encodeImg1], encodeImg2)
print(sonuclar, faceDis)
cv2.putText(fotograf2, f'{sonuclar} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Fotograf 1', fotograf1)
cv2.imshow('Fotograf 2', fotograf2)
cv2.waitKey(0)
