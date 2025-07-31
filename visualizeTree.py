import pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
from pathlib import Path

# === 1. Modell und Daten laden ===
model_path = Path().cwd() / 'trainedModels' / 'XGBBoostUnsimplified.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

data_path = Path().cwd() / 'finishedData' / 'train_test_split.npz'
data = np.load(data_path)
XTest = data['XTest']
yPred = model.predict(XTest)

# === 2. Surrogate Tree trainieren ===
surrogate = DecisionTreeRegressor(max_depth=4, random_state=42)
surrogate.fit(XTest, yPred)

# === 3. Visualisierung mit matplotlib ===
plt.figure(figsize=(20, 15))
plot_tree(
    surrogate,
    filled=True,
    rounded=True,
    feature_names=[f"Feature {i}" for i in range(XTest.shape[1])]
)
plt.title('Surrogate Tree – Visualisierung des Boosting-Modells')
plt.show()
