import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

# --- Scénario adversarial : bruit de 5% sur les coordonnées ---
# Pourquoi ça aide : avec 4 points parfaitement aux coins {0,1}², le point (0,0)
# donne z1 = b1 quelle que soit W1 → gradient nul depuis ce point → symétrie
# bloquante → réseau coincé à 50%. Le bruit casse cette symétrie et débloque
# l'apprentissage.
np.random.seed(7)                                    # seed séparé pour le bruit
X_xor += np.random.randn(*X_xor.shape) * 0.05
print("Coordonnées XOR après bruit :")
print(np.round(X_xor, 3))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_loss_bce(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Architecture 2-2-1 : 2 entrées → 2 neurones cachés → 1 sortie
np.random.seed(42)
W1 = np.random.randn(2, 2) * 0.5  # [2, 2]
b1 = np.zeros(2)                   # [2,]
W2 = np.random.randn(2, 1) * 0.5  # [2, 1]
b2 = np.zeros(1)                   # [1,]

# lr=1.0 : même raisonnement que phase 2, lr=0.5 génère des pas trop petits
# pour sortir du minimum local sur un si petit dataset.
learning_rate = 0.5
n_epochs = 10000
losses = []

for epoch in range(n_epochs):
    # Forward pass couche 1 (cachée)
    z1 = X_xor @ W1 + b1      # [4, 2]
    a1 = sigmoid(z1)           # [4, 2]

    # Forward pass couche 2 (sortie)
    z2 = a1 @ W2 + b2          # [4, 1]
    a2 = sigmoid(z2)           # [4, 1]
    y_pred = a2.flatten()      # [4,]

    loss = compute_loss_bce(y_xor, y_pred)
    losses.append(loss)

    # Backprop couche 2 — même formule simplifiée (BCE+sigmoid)
    error2 = y_pred - y_xor                           # [4,]
    dW2 = a1.T @ error2.reshape(-1, 1) / len(y_xor)  # [2, 1]
    db2 = np.mean(error2)

    # Backprop couche 1 — chain rule via W2 et dérivée sigmoid : σ'(x) = σ(x)*(1-σ(x))
    error1 = (error2.reshape(-1, 1) @ W2.T) * a1 * (1 - a1)  # [4, 2]
    dW1 = X_xor.T @ error1 / len(y_xor)                       # [2, 2]
    db1 = np.mean(error1, axis=0)                              # [2,]

    # Mise à jour
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 2000 == 0:
        acc = np.mean((y_pred > 0.5) == y_xor)
        print(f"Epoch {epoch:5d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

# Frontière de décision
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
z1g = sigmoid(np.dot(grid, W1) + b1)
z2g = sigmoid(np.dot(z1g, W2) + b2).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, z2g, alpha=0.4, cmap='RdBu')
plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, s=100, cmap='RdBu', edgecolors='k')
plt.title("XOR : frontière de décision du réseau 2-2-1")
plt.savefig("phase3_xor_boundary.png", dpi=100, bbox_inches='tight')
print(f"\nLoss finale : {losses[-1]:.4f}")
print(f"Accuracy finale : {np.mean((y_pred > 0.5) == y_xor):.2%}")