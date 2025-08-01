Benedikt hat das 6-Layer-CNN (own_cnn.ipynb) erstellt und trainiert, während Janik das resnet18 (notebook.ipynb/notebook_simplified.ipynb) Modell erstellt und trainiert hat. Benedikt hat dann das finale notebook.ipynb so angepasst, dass es
für beide Ansätze angewendet werden kann und Janik hat den Code für die Live-Demo im scripts-Ordner erstellt.

Notebooks Ordner enthält die Notebooks die zur Modellerstellung und Training benutzt wurden. Dabei ist notebook.ipynb das finale Notebook. Own_cnn beinhaltet die Entwicklung des 6-Layer-CNN und notebook_simplified beinhaltet Tests mit dem resnet18-Modell und dem simplen Datensatz

Models Ordner enthält zwischengespeicherte Modelle und Metriken, die geladen werden können um das Modell nicht bei jedem Durchgang neu trainieren zu müssen.

images enthält Grafiken zur Modellanalyse und Evaluierung

cv_results enthält die Ergebnisse der Grid_search_cv des 6-Layer-CNN

scripts enthält den Code für die Implementierung unseres Modells mit frontend etc.