'''Diese Datei wurde zur einfacheren Wiederverwendung der Bildfunktionen in den unterschiedlichen Notebooks 
erstellt. Die enthaltenen Funktionen wurden größtenteils im NotebookModelTraining erstellt und hierher kopiert.
Die Funktion zum Ausfiltern der Bilder wurde im filterBadPictures Notebook iterativ erarbeitet und erstellt.'''

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

'''Einlesen einer Grayscaleversion des jeweiligen Bilds'''
def readImage(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img

'''Anwendung des CLAHE Algorithmus auf das jeweilige Bild'''
def preprocessImage(img, size=(150,150)):
    #img = cv2.GaussianBlur(img,(5,5),0)
    #img = cv2.medianBlur(img,3)
    #img = cv2.bilateralFilter(img, 5, 75, 75)

    #img = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    img = cv2.resize(img, size)
    return img

'''Featureextraktion aus dem Bild und Erstellung der Featurearray daraus.'''
def processImage(img):
    # Feature 1: einfache Grayscale
    grayFlat = img.flatten()

    # Feature 2: sobel
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobel = cv2.magnitude(sobelx, sobely)
    sobelFlat = sobel.flatten()

    # Feature 3: Histogramm
    hist = cv2.calcHist([img], [0], None, [16], [0,256])
    hist = cv2.normalize(hist,hist).flatten()

    # Feature 4: LBP (Local Binary Pattern)
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    #lbp_hist, _ = np.histogram(lbp, bins = np.arange(0,10), range= (0,9), density = True)
    lbp = lbp.flatten()

    features = np.concatenate([grayFlat,sobelFlat,hist,lbp])
    return features

'''Die fertige Funktion zur Erkennung von Hinterfrundanteilen in Bildern. Der genaue Ablauf dessen ist im Bericht beschrieben.'''
def detect_background_presence(
    gray, k=7, morph_kernel_size=5, intensity_threshold=90, min_area=500, visualize=False
):
    h, w = gray.shape
    pixels = gray.reshape((-1, 1)).astype(np.float32)

    gmm = GaussianMixture(n_components=k, covariance_type="tied", max_iter=100, random_state=0)
    labels = gmm.fit_predict(pixels).reshape(h, w)

    bright_clusters = [
        i for i in range(k)
        if np.sum(labels == i) > 0 and 
           (gray[labels == i] > intensity_threshold).mean() > 0.8
    ]

    bright_mask = np.isin(labels, bright_clusters).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    cleaned_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask)
    filtered_mask = np.zeros_like(cleaned_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels_cc == i] = 255

    mask = filtered_mask.astype(bool)
    touches_top = np.any(mask[0, :])
    touches_bottom = np.any(mask[-1, :])

    # Accept only if it touches one (not both) of top or bottom
    has_valid_background = np.any(mask) and (touches_top ^ touches_bottom)

    if visualize:
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if has_valid_background:
            overlay[mask] = [0, 0, 255]  # highlight in red
        plt.figure(figsize=(10, 4))
        plt.imshow(overlay)
        plt.title("Detected Background (Red)" if has_valid_background else "No Valid Background")
        plt.axis("off")
        plt.show()

    return has_valid_background

'''Die folgenden zwei Funktionen werden genutzt, um die Visualisierung zu erstellen. process_patch stellt dabei eine Pipeline dar, die einen gewissen Bildausschnitt übernimmt,
eine Vorhersage auf diesen trifft mit allen dafür nötigen zwischenschritten und das Ergebnis dessen zurückgibt. Jeder Patch wird im Verlauf dessen hochskaliert,
um eine verarbeitbare Featurearray bei kleineren Ausschnitten zu erhalten.'''
def process_patch(args):
    y,x,patch,model = args
    preprocessed = preprocessImage(patch)
    features = processImage(preprocessed).reshape(1,-1)
    prob = model.predict_proba(features)[0][1]
    return(y,x,prob)

'''visualizeDecision ist die gesamtfunktion, um die Visualisierung durchzuführen.
Der ablauf ist dabei so, dass das Bild in sich überlappende Patches aufgeteilt wird, für die jeweils eine Vorhersage getroffen wird.
Die Einfärbung der jeweiligen Bildbereiche wird dann anhand der Anzahl von Pittingklassifikationen im jeweiligen Bereich getätigt. 
Ist also ein Teil vom Bild öfter an einer Pittingklassifizierung auf einen größeren Bereich beteiligt, so wird dieser als kritischer dargestellt.'''
def visualizeDecision (image, model, nameModel, pca, scaler, windowSize=30, stepSize=10):
    image = cv2.resize(image,(150,150))
    h,w = image.shape
    heatmap = np.zeros((h,w))
    countmap = np.zeros((h,w))

    for y in range(0,h-windowSize+1, stepSize):
        for x in range(0, w-windowSize+1, stepSize):
            patch = image[y:y+windowSize, x:x+windowSize]
            preprocessed = preprocessImage(patch)
            features = featureProcessing(processImage(preprocessed),pca,scaler)

            prob = model.predict_proba(features)[0][1]

            heatmap[y:y+windowSize, x:x+windowSize] += prob
            countmap[y:y+windowSize, x:x+windowSize] += 1

    heatmap/= (countmap + 1e-5)

    plt.imshow(image, cmap='gray')
    plt.imshow(heatmap, cmap='jet_r', alpha=0.5)
    plt.colorbar(label="Pitting probability")
    plt.title (f"Pitting Heatmap from {nameModel}")
    plt.show()

def featureProcessing (features, pca, scaler):
    features = features.reshape(1,-1)
    featuresScaled = scaler.transform(features)
    return pca.transform(featuresScaled)