from Bio import LogisticRegression
xs = [[-53, -200.78], [117, -267.14], [57, -163.47], [16, -190.30],
      [11, -220.94], [85, -193.94], [16, -182.71], [15, -180.41],
      [-26, -181.73], [58, -259.87], [126, -414.53], [191, -249.57],
      [113, -265.28], [145, -312.99], [154, -213.83], [147, -380.85],[93, -291.13]]
ys = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
model = LogisticRegression.train(xs, ys)

model.beta

def show_progress(iteration, loglikelihood):
    print("Iteration:", iteration, "Log-likelihood function:", loglikelihood)

model = LogisticRegression.train(xs, ys, update_fn=show_progress)

print("yxcE, yxcD:", LogisticRegression.classify(model, [6, -173.143442352]))
print("yxiB, yxiA:", LogisticRegression.classify(model, [309, -271.005880394]))

q, p = LogisticRegression.calculate(model, [6, -173.143442352])
print("class OP: probability =", p, "class NOP: probability =", q)

q, p = LogisticRegression.calculate(model, [309, -271.005880394])
print("class OP: probability =", p, "class NOP: probability =", q)

for i in range(len(ys)):
    print("True:", ys[i], "Predicted:", LogisticRegression.classify(model, xs[i]))

for i in range(len(ys)):
    print("True:", ys[i], "Predicted:", LogisticRegression.classify(model, xs[i]))

from Bio import kNN
k = 3
model = kNN.train(xs, ys, k)

x = [6, -173.143442352]
print("yxcE, yxcD:", kNN.classify(model, x))

x = [309, -271.005880394]
print("yxiB, yxiA:", kNN.classify(model, x))

def cityblock(x1, x2):
    assert len(x1)==2
    assert len(x2)==2
    distance = abs(x1[0]-x2[0]) + abs(x1[1]-x2[1])
    return distance

x = [6, -173.143442352]
print("yxcE, yxcD:", kNN.classify(model, x, distance_fn = cityblock))

from math import exp
def weight(x1, x2):
    assert len(x1)==2
    assert len(x2)==2
    return exp(-abs(x1[0]-x2[0]) - abs(x1[1]-x2[1]))

x = [6, -173.143442352]
print("yxcE, yxcD:", kNN.classify(model, x, weight_fn = weight))

x = [6, -173.143442352]
weight = kNN.calculate(model, x)
print("class OP: weight =", weight[0], "class NOP: weight =", weight[1])

x = [117, -267.14]
weight = kNN.calculate(model, x)
print("class OP: weight =", weight[0], "class NOP: weight =", weight[1])

for i in range(len(ys)):
    print("True:", ys[i], "Predicted:", kNN.classify(model, xs[i]))

for i in range(len(ys)):
    model = kNN.train(xs[:i]+xs[i+1:], ys[:i]+ys[i+1:], 3)
    print("True:", ys[i], "Predicted:", kNN.classify(model, xs[i]))



