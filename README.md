# Yüz Tanıma Sistemi
Sistemi kullanmak için ilk olarak bu repo'yu klonlayın<br>
`git clone https://github.com/SuleymanKaya1/Yuz-Tanima-Sistemi.git`<br>
<br>
Ardından klasöre geçiş yapın ve gerekli kütüphaneleri kurun<br>
`cd Yuz-Tanima-Sistemi && sudo apt install cmake g++ && pip3 install -r gereksinimler.txt`<br>
<br>
## Personel algılama nasıl yapılır?<br>
1 - Personellerin vesikalık fotoğrafını çekin<br>
2 - Çektiğiniz fotoğrafları kişilerin ismiyle yeniden adlandırın<br>
3 - Fotoğrafları 'PersonelAlgilama' klasörünün içine atın<br>
<br>
Örnek olarak X kişisini algılayalım:<br>
--> X kişisinin vesikalık fotoğrafını çekin<br>
--> Çektiğiniz fotoğrafı X.jpg olarak yeniden adlandırın.<br>
--> Bu fotoğrafı 'PersonelAlgilama' isimli klasöre aktarın<br>
<br>
Ardından yazılımı çalıştırın<br>
`python3 KameradanPersonelAlgilama.py`<br>
<br>
Yazılımı çalıştırdığınızda hangi personeli saat kaçta algıladığını Kayıtlar.csv'ye kaydedecektir. Bu sayede personellerin kurumunuza giriş-çıkış saatlerinin kayıtlarına ulaşabilirsiniz.
<br>
## Yüz karşılaştırma nasıl yapılır?<br>
1 - Karşılaştırmak istediğiniz fotoğrafları 'fotograf1.jpg' ve 'fotograf2.jpg' olarak yeniden adlandırın<br>
2 - 'YuzKarsilastirmaKlasoru' isimli klasörün içine atın.<br>
<br>
Ardından yüz karşılaştırma yazılımını çalıştırın<br>
`python3 YuzKarsilastirma.py`
<br>
<br>
#### Ubuntu 20.04.02'de test edilmiştir, hatasız çalışmaktadır.
