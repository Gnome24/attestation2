import numpy as np

# истинные значения
y_true = ['C','C','C','C','C','C', 'F','F','F','F','F','F','F','F','F','F', 'H','H','H','H','H','H','H','H','H']
# результат системы
y_pred = ['C','C','C','C','H','F', 'C','C','C','C','C','C','H','H','F','F', 'C','C','C','H','H','H','H','H','H']

classes = ['C', 'F', 'H']
confusion_matrix = np.zeros((3, 3))

for t, p in zip(y_true, y_pred):
    i = classes.index(t)
    j = classes.index(p)
    confusion_matrix[i, j] += 1

precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
f1_score = 2 * precision * recall / (precision + recall)

print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1_score)
