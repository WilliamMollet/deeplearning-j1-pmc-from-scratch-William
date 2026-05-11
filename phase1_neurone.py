import numpy as np

X = np.array([
    [0.2, 0.1],
    [0.8, 0.9],
    [0.3, 0.7],
    [0.9, 0.2],
])
y = np.array([0, 1, 1, 0])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)


def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Poids fixés — pas encore d'entraînement dans cette phase
w = np.array([0.5, -0.3])
b = 0.1

y_pred = forward(X, w, b)
loss = compute_loss(y, y_pred)

print(f"Prédictions : {y_pred.round(3)}")
print(f"Étiquettes  : {y}")
print(f"Loss BCE    : {loss:.4f}")