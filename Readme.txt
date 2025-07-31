Dieser Ordner enthält alle Codefiles und bearbeiteten Daten der klassischen Modelle (Gradient Boosting und Random Forest).
Nicht enthalten sind die Datensätze. Diese sind um den Code einwandfrei nutzen zu können in einem Ordner auf gleicher Ebene wie der pitting_detection
ordner abzulegen. Die Namen der enthaltenen Ordner wurden nicht angepasst.

Da die Dateien innerhalb der enthaltenen Ordner sind für die Abgabe zu groß, daher sind diese über folgenden Link zum Download verfügbar:


Die Files NotebookModelTraining und NotebookTrainedModels sind besonders wichtig und enthalten den Großteil des Codes.
filterBadPictures war lediglich die Datei, in der der Algorithmus zur Hintergrunderkennung erarbeitet wurde. 
imageFunctions enthält die genutzten Bildfunktionen. Wichtig ist diese Datei, da sie es erleichtert, die Funktionen wiederzuverwenden.
Dies ist zwar aus einem jupyter Notebook möglich, aber weniger einfach.

VisualizeDataChanges und visualizeTree sind Dateien, die nicht zur eigentlichen Bearbeitung der Problemstellung genutzt werden.
Hier wurden nur Bilder für die Präsentation erstellt. Der vollständigkeit halber werden diese jedoch mit abgegeben.

Der Ordner finishedData enthält für die Verarbeitung wichtige Dateien, wie den fertigen Pca scaler und den fertig verarbeiteten Datensatz. Ebenso
ist ein textdokument mit den gefilterten Hintergrundbildern enthalten. Ersichtlich ist, dass nicht nur Hintergrundbilder gefiltert wurden.
resultVideos enthält die beiden erstellten Videos mit klassifikation auf das gegebene Video. Die Verarbeitung war dabei nicht in realtime möglich.
trainedModels enthält die auf den ganzen Datensatz angepassten Modelle.