import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from skimage import io, color, transform
from skimage.feature import hog
from skimage.util import img_as_ubyte  # Utilisé pour assurer la compatibilité du format d'image


# Charger et prétraiter les images
def load_images(dataset_dir):
    X = []
    y = []

    for label in ['cat', 'dog']:
        folder = os.path.join(dataset_dir, label)

        # Vérifier si le dossier existe
        if not os.path.exists(folder):
            print(f"Dossier {folder} non trouvé.")
            continue

        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)

            # Vérifier si le fichier est une image
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            try:
                image = io.imread(img_path)
                image = color.rgb2gray(image)  # Convertir en niveau de gris
                image = transform.resize(image, (64, 64))  # Redimensionner l'image
                image = img_as_ubyte(image)  # Convertir en format entier non signé 8 bits (compatible HOG)

                # Extraire les caractéristiques HOG
                features, _ = hog(image, block_norm='L2-Hys',
                                  pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                  visualize=True)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"Erreur lors de la lecture de l'image {img_path}: {e}")
                continue

    X = np.array(X)
    y = np.array(y)
    return X, y


# Charger le dataset
dataset_dir = 'dataset'
X, y = load_images(dataset_dir)

# Vérifier si des images ont été chargées
if X.shape[0] == 0:
    print("Aucune image n'a été chargée. Vérifiez le chemin du dataset.")
else:
    # Encoder les étiquettes (cat -> 0, dog -> 1)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer et entraîner le modèle SVM avec un noyau linéaire
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calculer l'accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy du modèle SVM: {accuracy * 100:.2f}%")
