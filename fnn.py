import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sn
# Function to plot the confusion matrix (code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)
from sklearn import metrics
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


data = pd.read_json('../yelp_dataset/yelp_dataset_labeled.json', lines=True).fillna(0)


y = data['label']
X = data.drop(columns=['label', 'business_id'], axis=1)
X['city'] = LabelEncoder().fit_transform(X['city'])
X['state'] = LabelEncoder().fit_transform(X['state'])
scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# y = data['label']
# city = data['city']
# state = data['state']
# city_dummies = pd.get_dummies(city, drop_first=True)
# state_dummies = pd.get_dummies(state, drop_first=True)
# x = pd.concat(objs=[data.drop(columns=['business_id', 'label', 'city', 'state']), city_dummies, state_dummies], axis='columns')
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

model = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ]
)

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

batch_size = 32
epochs = 30

history = model.fit(
    X_train,
    y_train,
    validation_split = 0.2,
    batch_size = batch_size,
    epochs = epochs,
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau()]
)


plt.figure(figsize=(14, 10))

epochs_range = range(epochs)
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs_range, train_loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Time")
plt.legend()

plt.show()

model.evaluate(X_test, y_test)
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(model.predict(X_test), axis=-1)

cm = metrics.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
print(classification_report(y_test, y_pred))
print('ml train model auc score {:.6f}'.format(roc_auc_score(y_test, y_pred_prob)))