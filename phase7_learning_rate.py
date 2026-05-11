import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test  = X_test.reshape(-1, 784).astype('float32') / 255.0

learning_rates = [1e-7, 1e-3, 1.0]
lr_labels = ['trop petit (1e-7)', 'sweet spot (1e-3)', 'trop grand (1.0)']

results = []
histories = {}

for lr, label in zip(learning_rates, lr_labels):
    tf.random.set_seed(42)

    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(64,  activation='relu'),
        keras.layers.Dense(10,  activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=0
    )
    elapsed = time.time() - start

    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    val_losses = history.history['val_loss']

    results.append({
        'lr': lr,
        'label': label,
        'val_loss_final': val_losses[-1],
        'test_accuracy': test_acc,
        'train_time_s': elapsed,
    })
    histories[label] = val_losses
    print(f"lr={lr:.0e} | {label:24s} | val_loss: {val_losses[-1]:.4f} | test_acc: {test_acc:.4f}")

# Tableau récapitulatif
print("\n=== TABLEAU COMPARATIF LEARNING RATE ===")
print(f"{'LR':8s} | {'Label':24s} | {'Val loss final':14s} | {'Test acc':10s} | {'Temps (s)':10s}")
print("-" * 80)
for r in results:
    print(f"{r['lr']:.0e} | {r['label']:24s} | {r['val_loss_final']:.4f}          | {r['test_accuracy']:.4f}     | {r['train_time_s']:.0f}")

# Courbe superposée (échelle log pour mieux voir les extrêmes)
plt.figure(figsize=(10, 5))
for label, val_losses in histories.items():
    plt.plot(range(1, 11), val_losses, label=label, linewidth=2)
plt.xlabel("Epoch"); plt.ylabel("Val Loss")
plt.title("Impact du learning rate sur la convergence (MNIST)")
plt.legend()
plt.yscale('log')
plt.savefig("phase7_lr_curve.png", dpi=100, bbox_inches='tight')
print("\nCourbe sauvegardée : phase7_lr_curve.png")