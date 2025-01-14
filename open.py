import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from skimage import io, color, transform
from skimage.feature import hog
# Charger et prétraiter les images
def load_images(dataset_dir):
    X = []
    y = []
    for label in ['cat', 'dog']:
        folder = os.path.join(dataset_dir, label)
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            image = io.imread(img_path)
            image = color.rgb2gray(image) # Convertir en niveau de gris
            image = transform.resize(image, (64, 64)) # Redimensionner l'image
            # Extraire les caractéristiques HOG (Histogram of OrientedGradients)
            features, _ = hog(image, block_norm='L2-Hys',
            pixels_per_cell=(8, 8), cells_per_block=(2, 2),
            visualize=True)
            X.append(features)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y

# Charger le dataset
dataset_dir = 'dataset'
X, y = load_images(dataset_dir)

# Encoder les étiquettes (cat -> 0, dog -> 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# Créer et entraîner le modèle SVM avec un noyau linéaire
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy du modèle SVM: {accuracy * 100:.2f}%")