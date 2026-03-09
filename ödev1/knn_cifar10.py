import numpy as np
import os
import cv2
import pandas as pd
import urllib.request
import tarfile
import pickle

class KNearestNeighbor:
    """
    K-En Yakın Komşu (K-NN) Sınıflandırıcısı.
    Bu sınıf AÇIK MATEMATİKSEL işlemlere (L2-Öklid uzaklık denklemleri üzerinden)
    dayalı olarak hesaplama yapar ve sınıflandırma gerçekleştirir.
    """
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        # KNN algoritmasında "eğitim" aşaması veriyi hafızaya almaktan ibarettir.
        self.X_train = X
        self.y_train = y

    def compute_distances(self, X):
        """
        AÇIK MATEMATİKSEL İŞLEM: (L2 - Öklid Uzaklığı)
        İki vektör (x ve y) arasındaki Öklid uzaklığının karesi cebirsel olarak şöyledir:
        d^2(x, y) = (x - y)^2 = x^2 + y^2 - 2xy
        
        Döngü (for) kullanmak tüm veri setinde çok uzun sürer.
        O yüzden bu temel matematiksel denklemi açık matris hesabına çevirdik:
        d^2 = sum(X^2) + sum(X_train^2) - 2 * (X dot X_train^T)
        Son olarak karekökünü alıyoruz: d = sqrt(d^2)
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        # 1. MATEMATİKSEL İŞLEM: x^2 (Test verisinin elemanlarının karelerinin toplamı)
        # Bütün px^2'lerin toplamı
        test_sum_sq = np.sum(np.square(X), axis=1)

        # 2. MATEMATİKSEL İŞLEM: y^2 (Eğitim verisinin elemanlarının karelerinin toplamı)
        train_sum_sq = np.sum(np.square(self.X_train), axis=1)

        # 3. MATEMATİKSEL İŞLEM: 2xy (İki veri kümesinin matris çarpımının 2 katı)
        two_xy = 2 * np.dot(X, self.X_train.T)

        # 4. AÇIK DENKLEMİN BİRLEŞTİRİLMESİ: sqrt(x^2 + y^2 - 2xy)
        # Matematiksel sıfırın altına inmemesi için np.maximum kullanılarak değer 0'da sınırlandırılır
        dists = np.sqrt(np.maximum(test_sum_sq[:, np.newaxis] + train_sum_sq - two_xy, 0.0))

        return dists

    def predict_labels(self, dists, k=1):
        """
        Matematiksel olarak elde edilen uzaklıklara dayanarak tahmin gerçekleştirir.
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        
        for i in range(num_test):
            # 1. Her test noktasının tüm eğitim kümesine olan sıralı uzaklıkları
            distances_for_i = dists[i, :]
            
            # En küçük (en yakın) uzaklıklara sahip k adet noktanın indisleri
            closest_indices = np.argsort(distances_for_i)[:k]
            
            # Bu indislerin etiket (sınıf) karşılıkları
            closest_y = self.y_train[closest_indices]
            
            # Majority Voting (Çoğunluk oylaması) - Frekansı en yüksek olan sınıf seçilir
            values, counts = np.unique(closest_y, return_counts=True)
            y_pred[i] = values[np.argmax(counts)]

        return y_pred


def load_cifar10_batch(file):
    """Cifar10 veri batch'ini pickle üzerinden okur"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = dict[b'data']
    Y = dict[b'labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    Y = np.array(Y)
    return X, Y

def fetch_and_load_cifar10_standalone():
    """
    Gerekirse CIFAR-10 verisetini (python sÃ¼rÃ¼mÃ¼) indirir, aÃ§ar ve belleğe yÃ¼kler.
    Ä°nternetten indirilen 'cifar-10-batches-py' standart Ã¶ğrenme dosyasÄ±dÄ±r.
    """
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_filename = "cifar-10-python.tar.gz"
    data_dir = "cifar-10-batches-py"
    
    if not os.path.exists(data_dir):
        if not os.path.exists(tar_filename):
            print(f"Standart Cifar-10 veriseti indiriliyor ({url}) ... Lütfen bekleyiniz...")
            urllib.request.urlretrieve(url, tar_filename)
            print("İndirme tamamlandı.")
        
        print("Tar arşivi çıkartılıyor...")
        with tarfile.open(tar_filename, "r:gz") as tar:
            tar.extractall()
    
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(data_dir, 'data_batch_%d' % (b, ))
        X, Y = load_cifar10_batch(f)
        xs.append(X)
        ys.append(Y)
    
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    
    X_test, y_test = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    
    return X_train, y_train, X_test, y_test

def load_dataset():
    """
    Kulanıcının yüklediği 'cifar-10/train' klasöründen açık resimleri (png) yüklemeyi dener.
    Eğer bulamazsa, standart Cifar10 indirme modülüne yönlendirerek indirip yükler.
    """
    dataset_dir = "cifar-10"
    train_dir = os.path.join(dataset_dir, "train")
    labels_file = os.path.join(dataset_dir, "trainLabels.csv")

    if os.path.exists(train_dir) and os.path.exists(labels_file):
        print("Mevcut yerel açık 'train' klasörü (%s) bulundu, açık dosyalar okunuyor..." % train_dir)
        df = pd.read_csv(labels_file)
        
        classes = sorted(list(df['label'].unique()))
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        print(f"Hedef sınıflar: {classes}")
        
        # Sadece 10.000 resim çekilir ki süper uzun sürmesin
        num_images = min(10000, len(df))
        X = []
        y = []
        
        for i in range(num_images):
            img_id = df.iloc[i]['id']
            label_str = df.iloc[i]['label']
            img_path = os.path.join(train_dir, f"{img_id}.png")
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # Renkleri opencv BGR varsayılanından RGB'ye al
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    X.append(img)
                    y.append(class_to_idx[label_str])
            
            if (i+1) % 2000 == 0:
                print(f"{i+1} veri resim olarak yüklendi...")
                
        X = np.array(X)
        y = np.array(y)
        
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_test, y_test
    else:
        print("Açık Cifar10 (.png) bulunamadı. Standart paket indiriliyor/yükleniyor...")
        return fetch_and_load_cifar10_standalone()

def main():
    print("=========================================================")
    print(" K-NN (K-En Yakın Komşu) Mimarisi (Açık Matematiksel Model) ")
    print("=========================================================")
    
    print("1) Veriseti hazırlanıyor...")
    X_train, y_train, X_test, y_test = load_dataset()
    print(f"Eğitim verisi (Train) kapasitesi: {X_train.shape}")
    print(f"Test verisi kapasitesi: {X_test.shape}")

    # Görüntü matrisleri işleyebilmek için 1 Boyutlu (Flatten) hale getiriliyor.
    # Optimizasyon bulma süresini makul kılmak adına alt set çekiliyor (Standart akademik prosedür)
    num_training = 5000
    num_test = 500

    X_train_sub = X_train[:num_training]
    y_train_sub = y_train[:num_training]
    X_test_sub = X_test[:num_test]
    y_test_sub = y_test[:num_test]

    X_train_sub = np.reshape(X_train_sub, (X_train_sub.shape[0], -1)).astype('float64')
    X_test_sub = np.reshape(X_test_sub, (X_test_sub.shape[0], -1)).astype('float64')

    print("\n2) Model Parametreleştiriliyor...")
    classifier = KNearestNeighbor()
    classifier.train(X_train_sub, y_train_sub)

    print("\n3) Açık Denklemlerle Öklid Uzaklık Matrisi Hesaplanıyor (d^2 = x^2 + y^2 - 2xy)...")
    dists = classifier.compute_distances(X_test_sub)
    print(f"Uzaklık matrisi oluşturuldu: Beklenen Şekil (Test x Eğitim) => {dists.shape}")

    k_choices = [1, 3, 5, 8, 10, 15, 20, 50, 100]
    
    print("\n---------------------------------------------------------")
    print("4) Optimal (En İyi) K Değerinin Aranması...")
    print("Her denemede bir K değeri denenip sonucu ekrana yazdırılmaktadır:")
    print("---------------------------------------------------------")
    
    best_k = 1
    best_accuracy = -1.0
    
    for k in k_choices:
        y_test_pred = classifier.predict_labels(dists, k=k)
        
        num_correct = np.sum(y_test_pred == y_test_sub)
        accuracy = float(num_correct) / num_test
        
        print(f"Deneme: k = {k:<3} | Doğruluk (Accuracy) Oranı: %{accuracy * 100:.2f} ({num_correct}/{num_test} doğru)")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            
    print("\n---------------------------------------------------------")
    print(" SONUÇ ")
    print("---------------------------------------------------------")
    print(f"Tüm denemeler tamamlandı ve optimal K limit değere ulaştı.")
    print(f"==> Matematiksel Optimal K (En İyi Değer)  : {best_k}")
    print(f"==> Ulaşılan En Yüksek Oran (Test Acc) : %{best_accuracy * 100:.2f}")
    print("---------------------------------------------------------")

if __name__ == '__main__':
    main()
