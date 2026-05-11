import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# ---- 1. Chargement ----
data = load_breast_cancer()
X, y = data.data, data.target
print(f"Shape X : {X.shape} | Classes : {np.unique(y)} ({data.target_names})")

# ---- 2. Préprocessing ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# IMPORTANT : fitter le scaler sur X_train uniquement (évite le data leakage)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

n_features = X_train.shape[1]
print(f"Train : {X_train.shape} | Test : {X_test.shape}")

# ---- 3. Pipeline numpy from-scratch ----
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def bce_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

np.random.seed(42)
# Architecture : n_features → 16 → 8 → 1, initialisation He
W1 = np.random.randn(n_features, 16) * np.sqrt(2 / n_features)
b1 = np.zeros(16)
W2 = np.random.randn(16, 8) * np.sqrt(2 / 16)
b2 = np.zeros(8)
W3 = np.random.randn(8, 1) * np.sqrt(2 / 8)
b3 = np.zeros(1)

lr_np = 0.001
n_epochs_np = 200
numpy_losses = []
n = len(y_train)

for epoch in range(n_epochs_np):
    # Forward
    z1 = X_train @ W1 + b1;  a1 = relu(z1)
    z2 = a1 @ W2 + b2;       a2 = relu(z2)
    z3 = a2 @ W3 + b3;       y_pred = sigmoid(z3).flatten()

    loss = bce_loss(y_train, y_pred)
    numpy_losses.append(loss)

    # Backward
    err3 = y_pred - y_train
    dW3 = a2.T @ err3.reshape(-1, 1) / n;  db3 = np.mean(err3)

    err2 = (err3.reshape(-1, 1) @ W3.T) * relu_grad(z2)
    dW2 = a1.T @ err2 / n;  db2 = np.mean(err2, axis=0)

    err1 = (err2 @ W2.T) * relu_grad(z1)
    dW1 = X_train.T @ err1 / n;  db1 = np.mean(err1, axis=0)

    W1 -= lr_np * dW1;  b1 -= lr_np * db1
    W2 -= lr_np * dW2;  b2 -= lr_np * db2
    W3 -= lr_np * dW3;  b3 -= lr_np * db3

# Évaluation numpy
z1t = relu(X_test @ W1 + b1)
z2t = relu(z1t @ W2 + b2)
y_pred_test = sigmoid(z2t @ W3 + b3).flatten()
numpy_acc  = np.mean((y_pred_test > 0.5) == y_test)
numpy_loss_final = bce_loss(y_test, y_pred_test)
print(f"Numpy from-scratch | Loss finale : {numpy_loss_final:.4f} | Test accuracy : {numpy_acc:.4f}")

# ---- 4. Pipeline Keras ----
tf.random.set_seed(42)
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(n_features,)),
    keras.layers.Dense(8,  activation='relu'),
    keras.layers.Dense(1,  activation='sigmoid'),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

start = time.time()
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)
elapsed = time.time() - start

keras_loss, keras_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Keras              | Loss finale : {keras_loss:.4f} | Test accuracy : {keras_acc:.4f}")
print(f"Gain Keras vs Numpy : +{(keras_acc - numpy_acc) * 100:.1f} points de %")
print(f"Temps Keras : {elapsed:.1f}s")

# ---- 5. Comparaison agrégée ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Courbes de loss
axes[0].plot(numpy_losses, label=f'Numpy ({n_epochs_np} epochs)', color='orange')
axes[0].plot(history.history['loss'], label='Keras train (50 epochs)', color='steelblue')
axes[0].plot(history.history['val_loss'], label='Keras val (50 epochs)', color='steelblue', linestyle='--')
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss BCE")
axes[0].set_title("Courbes de loss comparées")
axes[0].legend()

# Barplot accuracy
bars = axes[1].bar(['Numpy', 'Keras'], [numpy_acc, keras_acc], color=['orange', 'steelblue'])
axes[1].set_ylim(0.8, 1.0)
axes[1].set_ylabel("Test Accuracy")
axes[1].set_title("Comparaison accuracy — Breast Cancer")
for bar, val in zip(bars, [numpy_acc, keras_acc]):
    axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.002, f"{val:.4f}", ha='center')

plt.tight_layout()
plt.savefig("phase8_comparaison.png", dpi=100, bbox_inches='tight')
print("Graphe sauvegardé : phase8_comparaison.png")