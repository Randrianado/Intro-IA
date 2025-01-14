import gzip
import numpy as np
import matplotlib.pyplot as plt

# Fonction pour charger les images MNIST
def load_mnist_images(filename):
 with gzip.open(filename, 'rb') as f:
  data = np.frombuffer(f.read(), np.uint8, offset=16)
  data = data.reshape(-1, 28, 28)
 return data

# Chemin vers les fichiers MNIST locaux
train_images_path = 'mnist_data/train-images-idx3-ubyte.gz'

# Charger les images MNIST
images = load_mnist_images(train_images_path)

# Afficher 20 images
num_images_to_display = 20
plt.figure(figsize=(10, 6))
for i in range(num_images_to_display):
 plt.subplot(4, 5, i + 1)
 plt.imshow(images[i], cmap='gray')
 plt.axis('off') # Supprimer les axes
 plt.title(f'Label: {i}') # Afficher le label de l'image
plt.tight_layout()
plt.show()
