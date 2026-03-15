# Almanzar Hernandez Karla Dennise
# 4351 ITI  G-18
# -----------------------------------------------------------
# Red Neuronal Multicapa (MLP)
# Problema: Detección de inconsistencia entre dos sensores redundantes (XOR)
# -----------------------------------------------------------

import math
import random

LR = 0.1
EPOCHS = 10000
H1 = 3
H2 = 2

X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,1,1,0]

def sigmoid(x): return 1/(1+math.exp(-x))
def dsigmoid(a): return a*(1-a)
def dot(v,w): return sum(vi*wi for vi,wi in zip(v,w))

W1 = [[random.random() for _ in range(H1)] for _ in range(2)]
b1 = [0.0 for _ in range(H1)]
W2 = [[random.random() for _ in range(H2)] for _ in range(H1)]
b2 = [0.0 for _ in range(H2)]
W3 = [random.random() for _ in range(H2)]
b3 = random.random()

history = []

for epoch in range(EPOCHS):
    total_error = 0.0
    for xi, target in zip(X, y):
        z1 = [dot(xi, [W1[0][j], W1[1][j]]) + b1[j] for j in range(H1)]
        a1 = [sigmoid(z) for z in z1]
        z2 = [dot(a1, [W2[i][j] for i in range(H1)]) + b2[j] for j in range(H2)]
        a2 = [sigmoid(z) for z in z2]
        z3 = sum(a2[j]*W3[j] for j in range(H2)) + b3
        out = sigmoid(z3)

        error = target - out
        total_error += error*error

        d_out = error * dsigmoid(out)
        dW3 = [d_out*a2j for a2j in a2]; db3 = d_out

        da2 = [d_out*W3j for W3j in W3]
        dz2 = [da2j*dsigmoid(a2j) for da2j,a2j in zip(da2,a2)]
        dW2 = [[dz2[j]*a1[i] for j in range(H2)] for i in range(H1)]
        db2 = dz2[:]

        da1 = [sum(dz2[j]*W2[i][j] for j in range(H2)) for i in range(H1)]
        dz1 = [da1i*dsigmoid(a1i) for da1i,a1i in zip(da1,a1)]
        dW1 = [[dz1[j]*xi[i] for j in range(H1)] for i in range(2)]
        db1 = dz1[:]

        for j in range(H2): W3[j] += LR*dW3[j]
        b3 += LR*db3
        for i in range(H1):
            for j in range(H2): W2[i][j] += LR*dW2[i][j]
        for j in range(H2): b2[j] += LR*db2[j]
        for i in range(2):
            for j in range(H1): W1[i][j] += LR*dW1[i][j]
        for j in range(H1): b1[j] += LR*db1[j]

    history.append(total_error)
    if epoch % 1000 == 0:
        print(f"Época {epoch:5d} - Error total: {total_error:.6f}")

print("\nResultados finales:")
for xi in X:
    z1 = [dot(xi, [W1[0][j], W1[1][j]]) + b1[j] for j in range(H1)]
    a1 = [sigmoid(z) for z in z1]
    z2 = [dot(a1, [W2[i][j] for i in range(H1)]) + b2[j] for j in range(H2)]
    a2 = [sigmoid(z) for z in z2]
    z3 = sum(a2[j]*W3[j] for j in range(H2)) + b3
    out = sigmoid(z3)
    print(f"Entrada: {xi} -> Predicción: {out:.4f} (umbral 0.5 => {int(out>=0.5)})")

with open("history_error.txt", "w", encoding="utf-8") as f:
    for e in history: f.write(f"{e}\n")
