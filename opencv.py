import cv2
import matplotlib.pyplot as plt
# Charger l'image
image_path = '1.jpg' # Remplacez par le chemin de votre image
image = cv2.imread(image_path)

# Traitement 1 : Conversion en niveaux de gris
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Traitement 2 : Redimensionnement
resized_image = cv2.resize(image, (200, 200))

# Traitement 3 : Application d'un filtre Gaussian
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Traitement 4 : Détection de contours
edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

# Affichage des résultats
images = [image, gray_image, resized_image, edges]
titles = ['Originale', 'Niveaux de gris', 'Redimensionnée', 'Contoursdétectés']
plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 2, i + 1)
    if len(images[i].shape) == 2: # Si l'image est en niveaux de gris
        plt.imshow(images[i], cmap='gray')
    else: # Si l'image est en couleur
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')
plt.show()