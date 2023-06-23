---
jupyter:
  colab:
    toc_visible: true
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

link: [https://colab.research.google.com/drive/1sZEyFqWfXtA8_TtctdNcJ8sZKeBI7Qtf?usp=sharing](https://colab.research.google.com/drive/1sZEyFqWfXtA8_TtctdNcJ8sZKeBI7Qtf?usp=sharing)

<div class="cell markdown" id="mEOaOiomZyVr">

Import libraries

</div>

<div class="cell code" execution_count="1" id="bXTzjBD1Z5Rd">

``` python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout,Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
```

</div>

<div class="cell markdown" id="J7lOvevWD96M">

# **Loading dataset**

</div>

<div class="cell markdown" id="-0Fp6pKaXAOX">

This dataset consists of the scanned images of brain of patient
diagnosed of brain tumour.

</div>

<div class="cell code" id="Lz9UdsLgA0G_">

``` python
!pip install -q kaggle

from google.colab import files

files.upload()
```

</div>

<div class="cell code" execution_count="3" id="gqiJi2huA_c7">

``` python
!mkdir ~/.kaggle

!cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json
```

</div>

<div class="cell code" execution_count="4"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="_UbscH1WBDQC" outputId="5747cfbd-3216-4a0b-92c9-3daaaaf84bb3">

``` python
! kaggle datasets download -d preetviradiya/brian-tumor-dataset
```

<div class="output stream stdout">

    Downloading brian-tumor-dataset.zip to /content
     90% 97.0M/107M [00:01<00:00, 72.5MB/s]
    100% 107M/107M [00:01<00:00, 82.7MB/s] 

</div>

</div>

<div class="cell code" id="VbeozAeyBN-C">

``` python
!unzip brian-tumor-dataset
```

</div>

<div class="cell code" execution_count="6" id="25fB4lhxRTE0">

``` python
import os
import pandas as pd


tumor_dir = r'Brain Tumor Data Set/Brain Tumor Data Set/Brain Tumor'
healthy_dir = r'Brain Tumor Data Set/Brain Tumor Data Set/Healthy'
dir_list = [tumor_dir, healthy_dir]
image_path = []
label = []

for i, j in enumerate(dir_list):
    images_list = os.listdir(j)
    for f in images_list:
        image_path.append(j + "/" + f)
        if i == 0:
            label.append('Cancer')
        else:
            label.append('Not Cancer')

data = pd.DataFrame({'image_path': image_path, 'label': label})
```

</div>

<div class="cell code" execution_count="7"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:206}"
id="46C10f-ETeWn" outputId="0b0bcca0-4969-46d8-a9e0-d52a5be45755">

``` python
data.head()
```

<div class="output execute_result" execution_count="7">

                                              image_path   label
    0  Brain Tumor Data Set/Brain Tumor Data Set/Brai...  Cancer
    1  Brain Tumor Data Set/Brain Tumor Data Set/Brai...  Cancer
    2  Brain Tumor Data Set/Brain Tumor Data Set/Brai...  Cancer
    3  Brain Tumor Data Set/Brain Tumor Data Set/Brai...  Cancer
    4  Brain Tumor Data Set/Brain Tumor Data Set/Brai...  Cancer

</div>

</div>

<div class="cell code" execution_count="8"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="tVa05RedXJxZ" outputId="e4c1c2d6-af86-4b91-ce87-b2d73ab3402e">

``` python
data.shape
```

<div class="output execute_result" execution_count="8">

    (4600, 2)

</div>

</div>

<div class="cell code" execution_count="9"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="DHHu08GKUob1" outputId="2653b3a1-a487-4f30-a6ef-4ffeabda3a06">

``` python
data['label'].value_counts()
```

<div class="output execute_result" execution_count="9">

    Cancer        2513
    Not Cancer    2087
    Name: label, dtype: int64

</div>

</div>

<div class="cell markdown" id="mz3xQxYpXhPg">

# **Split data into train, validation, test**

</div>

<div class="cell markdown" id="91Pc2i3fYx-H">

Split the data into train, validation and test sets with percentages of
80%, 10% and 10% respectively.

</div>

<div class="cell code" execution_count="10" id="lJcm1tNjYQmn">

``` python
from sklearn.model_selection import train_test_split
seed = 123

# Chia dữ liệu thành tập train và tập còn lại
train_set, remain_set = train_test_split(data, test_size=0.2, random_state=seed)

# Chia tập còn lại thành tập validation và tập test
val_set, test_set = train_test_split(remain_set, test_size=0.5, random_state=seed)
```

</div>

<div class="cell code" execution_count="11"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="QunnordMzrUl" outputId="4d4f6759-efc1-4f0c-d612-96d6107d814c">

``` python
print(train_set.shape)
print(val_set.shape)
print(test_set.shape)
```

<div class="output stream stdout">

    (3680, 2)
    (460, 2)
    (460, 2)

</div>

</div>

<div class="cell markdown" id="KPBswc9Xz6bu">

# **ImageDataGenerator**

</div>

<div class="cell code" execution_count="12"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="jAqmzn-c0EXx" outputId="8fbdfb16-4317-45bb-c2a0-67cc94bc3afc">

``` python
#Generate batches of tensor image data with real-time data augmentation.
image_generator = ImageDataGenerator(preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input)

train = image_generator.flow_from_dataframe(dataframe = train_set, x_col = "image_path", y_col ="label",
                                      target_size = (244, 244),
                                      color_mode = 'rgb',
                                      class_mode = "categorical",
                                      batch_size = 32,
                                      shuffle = False
                                     )
val = image_generator.flow_from_dataframe(dataframe = val_set, x_col ="image_path", y_col ="label",
                                    target_size=(244, 244),
                                    color_mode = 'rgb',
                                    class_mode = "categorical",
                                    batch_size = 32,
                                    shuffle = False
                                   )
test = image_generator.flow_from_dataframe(dataframe = test_set, x_col = "image_path", y_col ="label",
                                     target_size = (244, 244),
                                     color_mode = 'rgb',
                                     class_mode = "categorical",
                                     batch_size = 32,
                                     shuffle = False
                                    )
```

<div class="output stream stdout">

    Found 3680 validated image filenames belonging to 2 classes.
    Found 460 validated image filenames belonging to 2 classes.
    Found 460 validated image filenames belonging to 2 classes.

</div>

</div>

<div class="cell markdown" id="jX_VfnwOAUk9">

# **Visualize some images**

</div>

<div class="cell code" execution_count="13" id="V1TTP8Y-5fHw">

``` python
import matplotlib.pyplot as plt

def show_images(image_generator):
    img, label = image_generator.next()
    plt.figure(figsize=(20,20))
    for i in range(15):
        plt.subplot(5, 5, i+1)

        plt.imshow((img[i]+1)/2)  #scale images between 0 and 1

        idx = np.argmax(label[i])
        if idx == 0:
            plt.title('Cancer')
        else:
            plt.title('Not Cancer')
        plt.axis('off')
    plt.show()
```

</div>

<div class="cell code" execution_count="14"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:919}"
id="QHNwl4hdAGUw" outputId="6c18ac81-54bd-458d-9eae-f643c8b07c1c">

``` python
show_images(train)
```

<div class="output display_data">

![](10f89acc386637d81e76e108dc9d7b4c40573f3f.png)

</div>

</div>

<div class="cell markdown" id="Nor1UJxN0E7p">

# **Train Model**

</div>

<div class="cell code" execution_count="15"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="LI-R9tYGExfF" outputId="b196a62d-baa6-47cc-bfd0-981b6bb3a01d">

``` python
# Thiết lập Convolutional Neural Networks (CNN):

import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(244, 244, 3)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train, epochs=10, verbose=1, validation_data = val)
```

<div class="output stream stdout">

    Epoch 1/10
    115/115 [==============================] - 1311s 11s/step - loss: 3.1010 - accuracy: 0.6742 - val_loss: 0.3472 - val_accuracy: 0.8783
    Epoch 2/10
    115/115 [==============================] - 1284s 11s/step - loss: 0.2459 - accuracy: 0.9087 - val_loss: 0.1274 - val_accuracy: 0.9630
    Epoch 3/10
    115/115 [==============================] - 1277s 11s/step - loss: 0.0912 - accuracy: 0.9685 - val_loss: 0.0988 - val_accuracy: 0.9652
    Epoch 4/10
    115/115 [==============================] - 1332s 12s/step - loss: 0.0552 - accuracy: 0.9812 - val_loss: 0.0973 - val_accuracy: 0.9739
    Epoch 5/10
    115/115 [==============================] - 1292s 11s/step - loss: 0.0323 - accuracy: 0.9875 - val_loss: 0.1113 - val_accuracy: 0.9717
    Epoch 6/10
    115/115 [==============================] - 1311s 11s/step - loss: 0.0266 - accuracy: 0.9905 - val_loss: 0.1038 - val_accuracy: 0.9739
    Epoch 7/10
    115/115 [==============================] - 1287s 11s/step - loss: 0.0211 - accuracy: 0.9908 - val_loss: 0.1308 - val_accuracy: 0.9674
    Epoch 8/10
    115/115 [==============================] - 1283s 11s/step - loss: 0.0373 - accuracy: 0.9910 - val_loss: 0.1418 - val_accuracy: 0.9565
    Epoch 9/10
    115/115 [==============================] - 1355s 12s/step - loss: 0.0316 - accuracy: 0.9872 - val_loss: 0.1020 - val_accuracy: 0.9761
    Epoch 10/10
    115/115 [==============================] - 1304s 11s/step - loss: 0.0191 - accuracy: 0.9921 - val_loss: 0.1541 - val_accuracy: 0.9696

</div>

</div>

<div class="cell markdown" id="CIvA-4zI8YJa">

# Plot Accuracy and Loss

</div>

<div class="cell code" execution_count="16"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:747}"
id="Q7-WT4IV5lKH" outputId="aa1e9dda-85fd-47e0-cf3a-6fc71368dacc">

``` python
# Accuracy
acc = history.history["accuracy"] # report of model
val_acc = history.history["val_accuracy"] # history of validation data

plt.figure(figsize=(8,8))
plt.subplot(2,1,1) # 2 rows and 1 columns

plt.plot(acc,label="Training Accuracy")
plt.plot(val_acc, label="Validation Acccuracy")

plt.legend()
plt.ylabel("Accuracy", fontsize=12)
plt.title("Training and Validation Accuracy", fontsize=12)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Loss
loss = history.history["loss"]        # Training loss
val_loss = history.history["val_loss"] # validation loss

plt.figure(figsize=(8,8))
plt.subplot(2,1,2)

plt.plot(loss, label="Training Loss")      #Training loss
plt.plot(val_loss, label="Validation Loss") # Validation Loss

plt.legend()
plt.ylim([min(plt.ylim()),1])
plt.ylabel("Loss", fontsize=12)
plt.title("Training and Validation Losses", fontsize=12)
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()
```

<div class="output display_data">

![](79e527e1936032c822a8e04bd5d92f4474933e53.png)

</div>

<div class="output display_data">

![](ad08ba2d31ba475ff562d39f44c9fa23eb33c2db.png)

</div>

</div>

<div class="cell markdown" id="j5G81-_u7q5i">

# **Predict test set**

</div>

<div class="cell code" execution_count="17"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="AHrDLyf65tdH" outputId="8f891959-caf6-4b03-d44d-9e665f6e0825">

``` python
model.evaluate(test, verbose=1)
```

<div class="output stream stdout">

    15/15 [==============================] - 38s 3s/step - loss: 0.1230 - accuracy: 0.9717

</div>

<div class="output execute_result" execution_count="17">

    [0.1230454370379448, 0.9717391133308411]

</div>

</div>

<div class="cell code" execution_count="18"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="sxn6liQ751kx" outputId="56a59de1-1504-4f43-86c3-5f39a5734541">

``` python
pred = model.predict(test)
y_pred = np.argmax(pred, axis=1)
```

<div class="output stream stdout">

    15/15 [==============================] - 41s 3s/step

</div>

</div>

<div class="cell code" execution_count="19" id="BCmqXZv852pa">

``` python
y_test = test.labels
y_test = np.array(y_test)
```

</div>

<div class="cell code" execution_count="20"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="cqY7Anaz53pN" outputId="1ea39c03-2bc9-4ed4-d9cf-53503b34cc6d">

``` python
from sklearn.metrics import accuracy_score

print("Accuracy of the Model:",accuracy_score(y_test, y_pred)*100,"%")
```

<div class="output stream stdout">

    Accuracy of the Model: 97.17391304347827 %

</div>

</div>

<div class="cell code" execution_count="21"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:462}"
id="B-VAaLXm55Sj" outputId="ebd3cb3c-ebd8-4591-97a5-7384b4aeb074">

``` python
from sklearn.metrics import confusion_matrix, accuracy_score

plt.figure(figsize = (10,5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt = 'g', cmap = 'crest')
```

<div class="output execute_result" execution_count="21">

    <Axes: >

</div>

<div class="output display_data">

![](9459d6c3bee46365ec576c33ae0da88bbed7bfab.png)

</div>

</div>

<div class="cell markdown" id="RbAU6vCh8lK-">

# **Early Stopping**

</div>

<div class="cell markdown" id="cLiOINEQ9SZ4">

To save training time we can stop training the CNN if the accuracy of
the validation data does not improve after a certain number of steps.
For example, the selection criteria is Accuracy on Validation data and
the algorithm will stop after 5 steps.

</div>

<div class="cell code" execution_count="22"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="ar5sgSHe8nSX" outputId="2279e6a6-ce86-49fb-a0e8-c1be246496b4">

``` python
from tensorflow.keras.callbacks import EarlyStopping

model_2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(244, 244, 3)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 5, min_delta=1)
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_2 = model_2.fit(train, epochs=10, verbose=1, validation_data = val, callbacks=[early_stop])
```

<div class="output stream stdout">

    Epoch 1/10
    115/115 [==============================] - 1307s 11s/step - loss: 2.7671 - accuracy: 0.7293 - val_loss: 0.2682 - val_accuracy: 0.8891
    Epoch 2/10
    115/115 [==============================] - 1222s 11s/step - loss: 0.1555 - accuracy: 0.9438 - val_loss: 0.1146 - val_accuracy: 0.9565
    Epoch 3/10
    115/115 [==============================] - 1259s 11s/step - loss: 0.0406 - accuracy: 0.9875 - val_loss: 0.0893 - val_accuracy: 0.9630
    Epoch 4/10
    115/115 [==============================] - 1248s 11s/step - loss: 0.0246 - accuracy: 0.9929 - val_loss: 0.0933 - val_accuracy: 0.9696
    Epoch 5/10
    115/115 [==============================] - 1245s 11s/step - loss: 0.0118 - accuracy: 0.9962 - val_loss: 0.1103 - val_accuracy: 0.9652
    Epoch 6/10
    115/115 [==============================] - 1270s 11s/step - loss: 0.0141 - accuracy: 0.9962 - val_loss: 0.0959 - val_accuracy: 0.9739

</div>

</div>

<div class="cell code" execution_count="23"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="cHxZxrwp91xi" outputId="ddc70a72-e272-43ba-eda3-40524d7bb1d0">

``` python
print('Train Accuracy with Early Stopping:', model_2.evaluate(test, verbose=1))
```

<div class="output stream stdout">

    15/15 [==============================] - 36s 2s/step - loss: 0.0965 - accuracy: 0.9717
    Train Accuracy with Early Stopping: [0.09646687656641006, 0.9717391133308411]

</div>

</div>

<div class="cell code" execution_count="24"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:747}"
id="CVlah8rzB_S1" outputId="3bf2acd8-0954-4c26-9fb7-e8fa55b72665">

``` python
# Accuracy
acc = history_2.history["accuracy"] # report of model
val_acc = history_2.history["val_accuracy"] # history of validation data

plt.figure(figsize=(8,8))
plt.subplot(2,1,1) # 2 rows and 1 columns

plt.plot(acc,label="Training Accuracy")
plt.plot(val_acc, label="Validation Acccuracy")

plt.legend()
plt.ylabel("Accuracy", fontsize=12)
plt.title("Training and Validation Accuracy", fontsize=12)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Loss
loss = history_2.history["loss"]        # Training loss
val_loss = history_2.history["val_loss"] # validation loss

plt.figure(figsize=(8,8))
plt.subplot(2,1,2)

plt.plot(loss, label="Training Loss")      #Training loss
plt.plot(val_loss, label="Validation Loss") # Validation Loss

plt.legend()
plt.ylim([min(plt.ylim()),1])
plt.ylabel("Loss", fontsize=12)
plt.title("Training and Validation Losses", fontsize=12)
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()
```

<div class="output display_data">

![](9d12d7722bc478725b4316f35f16ff21495ee8bd.png)

</div>

<div class="output display_data">

![](1814f665f45d616635d83d3649976691d02afbdd.png)

</div>

</div>
