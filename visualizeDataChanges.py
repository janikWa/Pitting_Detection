import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.feature import local_binary_pattern

path = (Path().parent /'data'/'KGT_pitting'/'P (2).png ').resolve()
print("Lade Bild von:", path)

if not path.exists():
    raise FileNotFoundError(f"Bild nicht gefunden: {path}")

image = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1)
sobel = cv2.magnitude(sobelx, sobely)
hist = cv2.calcHist([image], [0], None, [16], [0,256])
hist = hist.flatten()
lbp = local_binary_pattern(image, P=8, R=1, method='uniform')

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Original
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title("Originalbild")
axs[0, 0].axis('off')

# Sobel
axs[0, 1].imshow(sobel, cmap='gray')
axs[0, 1].set_title("Sobel-Kantenbild")
axs[0, 1].axis('off')

# LBP
axs[1, 0].imshow(lbp, cmap='gray')
axs[1, 0].set_title("Local Binary Pattern (LBP)")
axs[1, 0].axis('off')

# Histogramm
axs[1, 1].bar(range(16), hist, color='black')
axs[1, 1].set_title("Histogramm (16 Grauwerte)")
axs[1, 1].set_xlabel("Grauwerte")
axs[1, 1].set_ylabel("Anzahl")

plt.tight_layout()
#plt.savefig("vergleichsbild.png", dpi=300)  # Bild speichern
plt.show()