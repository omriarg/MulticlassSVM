from Multi_class_svm import MultiClassSVM
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import numpy as np
import torch
torch.manual_seed(1)#18
# שלב 1: הורדת הדאטה העיקרי
data = yf.download('GOOGL', period='25y', interval='1d')
# שלב 2: יצירת המודל והגדרת ההיפר-פרמטרים שנמצאו כהטובים ביותר
model = MultiClassSVM(data)
# הרצת grid search/ניסוי בוצעה בנפרד:
# model.test_best_kernel_and_hyperparams(method='GD')

model_params = {#also storing training information
    'kernel_type': 'polynomial',
    'C': 0.3, 'degree': 2, 'const': 0,'alpha':1e-8
    ,'epochs':40000,'batch_size':32,'training_method':'GradientDescent'
}

model.create_classifiers(kernel_type=model_params['kernel_type'], params=model_params,num_classes=3)
# שלב 3: אימון על קבוצת האימון הראשית
X_train,Y_train=model.training_set
model.train(X_train,Y_train,epochs=40000,batch_size=32,alpha=model_params['alpha'],method='GD')#train with gradientDescent
# שלב 5: הערכה על test_set
print("\n--- Evaluation on model.test_set ---")
X_test, Y_test = model.test_set
preds_test = model.predict(X_test).numpy()
true_test = Y_test.numpy()

accuracy_test = (preds_test == true_test).mean()
print(f"Test set accuracy: {accuracy_test * 100:.2f}%")

cm_test = confusion_matrix(true_test, preds_test)
print("Test set Confusion Matrix:")
print(cm_test)

print("Test set Classification Report:")
print(classification_report(true_test, preds_test, digits=4))

plt.figure(figsize=(6, 4))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens',
            xticklabels=[f'Pred {i}' for i in range(5)],
            yticklabels=[f'True {i}' for i in range(5)])
plt.title('Confusion Matrix for model.test_set')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# שלב 6: הערכה על 5 מניות נוספות
stocks_paths = [
    os.path.join(os.getcwd(), 'Stocks', 'JNJ_X_full.csv'),
    os.path.join(os.getcwd(), 'Stocks', 'KO_X_full.csv'),
    os.path.join(os.getcwd(), 'Stocks', 'NVDA_X_full.csv'),
    os.path.join(os.getcwd(), 'Stocks', 'PEP_X_full.csv'),
    os.path.join(os.getcwd(), 'Stocks', 'MSFT_X_full.csv'),
]
data_5_stocks = [pd.read_csv(path) for path in stocks_paths]
print("\n--- Evaluation on 5 Stocks (combined) ---")

all_true_labels = []
all_preds = []

for stock_data in data_5_stocks:
    X, Y = model.data_processed.prepare_new_df(stock_data)

    preds = model.predict(X).numpy()
    true_labels = Y.numpy()

    all_true_labels.extend(true_labels)
    all_preds.extend(preds)

# המרת לרשימות רגילות ל-numpy arrays
all_true_labels = np.array(all_true_labels)
all_preds = np.array(all_preds)

accuracy = (all_preds == all_true_labels).mean()
print(f"Combined Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(all_true_labels, all_preds)
print("Combined Confusion Matrix:")
print(cm)

print("Combined Classification Report:")
print(classification_report(all_true_labels, all_preds, digits=4))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=[f'Pred {i}' for i in range(5)],
            yticklabels=[f'True {i}' for i in range(5)])
plt.title('Combined Confusion Matrix for 5 Stocks')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()