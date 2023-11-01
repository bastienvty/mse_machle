## Iteration 1
- Best Estimator: SVC(C=0.1, kernel='linear')
- Best Score: 0.901
- Classification Report:

```
              precision    recall  f1-score   support

           0       0.89      0.93      0.91       100
           1       0.88      0.99      0.93       100
           2       0.80      0.86      0.83       100
           3       0.89      0.85      0.87       100
           4       0.86      0.91      0.88       100
           5       0.89      0.86      0.87       100
           6       0.91      0.93      0.92       100
           7       0.91      0.93      0.92       100
           8       0.88      0.75      0.81       100
           9       0.89      0.80      0.84       100

    accuracy                           0.88      1000
   macro avg       0.88      0.88      0.88      1000
weighted avg       0.88      0.88      0.88      1000

```
- Confusion Matrix:

![Confusion Matrix - Iteration 1](images/confusion_matrix_1.png)

----------------------------------------------------------------------## Iteration 2
- Best Estimator: SVC(C=10, coef0=0, degree=2, kernel='poly')
- Best Score: 0.9279999999999999
- Classification Report:

```
              precision    recall  f1-score   support

           0       0.91      0.94      0.93       100
           1       0.90      1.00      0.95       100
           2       0.88      0.92      0.90       100
           3       0.96      0.86      0.91       100
           4       0.91      0.94      0.93       100
           5       0.90      0.87      0.88       100
           6       0.91      0.93      0.92       100
           7       0.96      0.96      0.96       100
           8       0.90      0.87      0.88       100
           9       0.91      0.85      0.88       100

    accuracy                           0.91      1000
   macro avg       0.91      0.91      0.91      1000
weighted avg       0.91      0.91      0.91      1000

```
- Confusion Matrix:

![Confusion Matrix - Iteration 2](images/confusion_matrix_2.png)

----------------------------------------------------------------------## Iteration 3
- Best Estimator: SVC(C=10, gamma=0.01)
- Best Score: 0.9345000000000001
- Classification Report:

```
              precision    recall  f1-score   support

           0       0.93      0.95      0.94       100
           1       0.90      0.99      0.94       100
           2       0.84      0.91      0.87       100
           3       0.95      0.89      0.92       100
           4       0.90      0.92      0.91       100
           5       0.91      0.91      0.91       100
           6       0.96      0.94      0.95       100
           7       0.96      0.95      0.95       100
           8       0.91      0.85      0.88       100
           9       0.94      0.88      0.91       100

    accuracy                           0.92      1000
   macro avg       0.92      0.92      0.92      1000
weighted avg       0.92      0.92      0.92      1000

```
- Confusion Matrix:

![Confusion Matrix - Iteration 3](images/confusion_matrix_3.png)

----------------------------------------------------------------------## Iteration 4
- Best Estimator: SVC(C=1, coef0=0, kernel='sigmoid')
- Best Score: 0.8614999999999998
- Classification Report:

```
              precision    recall  f1-score   support

           0       0.85      0.90      0.87       100
           1       0.81      0.95      0.87       100
           2       0.76      0.76      0.76       100
           3       0.78      0.76      0.77       100
           4       0.81      0.89      0.85       100
           5       0.83      0.78      0.80       100
           6       0.84      0.87      0.86       100
           7       0.89      0.88      0.88       100
           8       0.80      0.68      0.74       100
           9       0.83      0.73      0.78       100

    accuracy                           0.82      1000
   macro avg       0.82      0.82      0.82      1000
weighted avg       0.82      0.82      0.82      1000

```
- Confusion Matrix:

![Confusion Matrix - Iteration 4](images/confusion_matrix_4.png)

----------------------------------------------------------------------