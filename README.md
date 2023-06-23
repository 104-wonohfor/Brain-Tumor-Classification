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

link: [https://colab.research.google.com/drive/1ZbEAKEaCelZz_ypE-_dnxcUp8Ky8fHh4?usp=sharing](https://colab.research.google.com/drive/1ZbEAKEaCelZz_ypE-_dnxcUp8Ky8fHh4?usp=sharing)

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
id="_UbscH1WBDQC" outputId="c4ac4531-0317-4461-c799-f3ad07071fc7">

``` python
! kaggle datasets download -d preetviradiya/brian-tumor-dataset
```

<div class="output stream stdout">

    Downloading brian-tumor-dataset.zip to /content
     99% 106M/107M [00:04<00:00, 32.8MB/s]
    100% 107M/107M [00:04<00:00, 23.6MB/s]

</div>

</div>

<div class="cell code" id="VbeozAeyBN-C">

``` python
!unzip brian-tumor-dataset
```

</div>

<div class="cell code" execution_count="7" id="25fB4lhxRTE0">

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

<div class="cell code" execution_count="8"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:206}"
id="46C10f-ETeWn" outputId="dfdf7bc9-ed78-4e11-b418-ddb27ec8b24e">

``` python
data.head()
```

<div class="output execute_result" execution_count="8">

                                              image_path   label
    0  Brain Tumor Data Set/Brain Tumor Data Set/Brai...  Cancer
    1  Brain Tumor Data Set/Brain Tumor Data Set/Brai...  Cancer
    2  Brain Tumor Data Set/Brain Tumor Data Set/Brai...  Cancer
    3  Brain Tumor Data Set/Brain Tumor Data Set/Brai...  Cancer
    4  Brain Tumor Data Set/Brain Tumor Data Set/Brai...  Cancer

</div>

</div>

<div class="cell code" execution_count="9"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="tVa05RedXJxZ" outputId="0ee153c2-bfcf-4c56-cd49-55a884ea942c">

``` python
data.shape
```

<div class="output execute_result" execution_count="9">

    (4600, 2)

</div>

</div>

<div class="cell code" execution_count="10"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="DHHu08GKUob1" outputId="a8619f53-76ec-4fd7-dd18-b27a4183151b">

``` python
data['label'].value_counts()
```

<div class="output execute_result" execution_count="10">

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

<div class="cell code" execution_count="11" id="lJcm1tNjYQmn">

``` python
from sklearn.model_selection import train_test_split
seed = 123

# Chia dữ liệu thành tập train và tập còn lại
train_set, remain_set = train_test_split(data, test_size=0.2, random_state=seed)

# Chia tập còn lại thành tập validation và tập test
val_set, test_set = train_test_split(remain_set, test_size=0.5, random_state=seed)
```

</div>

<div class="cell code" execution_count="12"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="QunnordMzrUl" outputId="3e26e7d5-1293-4ebd-dbb9-d833747d7944">

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

<div class="cell code" execution_count="13"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="jAqmzn-c0EXx" outputId="4f6207a8-5992-41cd-b9f6-03222a11b378">

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

<div class="cell code" execution_count="14" id="V1TTP8Y-5fHw">

``` python
import matplotlib.pyplot as plt

def show_images(image_generator):
    img, label = image_generator.next()
    plt.figure(figsize=(20,20))
    for i in range(15):
        plt.subplot(3, 5, i+1)

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

<div class="cell code" execution_count="15"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:919}"
id="QHNwl4hdAGUw" outputId="761eed92-1c2c-43a7-fa94-a242c6229ffd">

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

<div class="cell code" execution_count="16"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="LI-R9tYGExfF" outputId="1a7826fb-b7cb-4d81-b6fb-66da0ff3dd75">

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
    115/115 [==============================] - 1336s 12s/step - loss: 2.1103 - accuracy: 0.7231 - val_loss: 0.2288 - val_accuracy: 0.9261
    Epoch 2/10
    115/115 [==============================] - 1308s 11s/step - loss: 0.1478 - accuracy: 0.9459 - val_loss: 0.1051 - val_accuracy: 0.9652
    Epoch 3/10
    115/115 [==============================] - 1307s 11s/step - loss: 0.0446 - accuracy: 0.9859 - val_loss: 0.0917 - val_accuracy: 0.9674
    Epoch 4/10
    115/115 [==============================] - 1296s 11s/step - loss: 0.0156 - accuracy: 0.9959 - val_loss: 0.1160 - val_accuracy: 0.9652
    Epoch 5/10
    115/115 [==============================] - 1322s 11s/step - loss: 0.0149 - accuracy: 0.9962 - val_loss: 0.1306 - val_accuracy: 0.9717
    Epoch 6/10
    115/115 [==============================] - 1340s 12s/step - loss: 0.0101 - accuracy: 0.9970 - val_loss: 0.1325 - val_accuracy: 0.9674
    Epoch 7/10
    115/115 [==============================] - 1331s 12s/step - loss: 0.0109 - accuracy: 0.9973 - val_loss: 0.1393 - val_accuracy: 0.9609
    Epoch 8/10
    115/115 [==============================] - 1337s 12s/step - loss: 0.0121 - accuracy: 0.9978 - val_loss: 0.1636 - val_accuracy: 0.9609
    Epoch 9/10
    115/115 [==============================] - 1330s 12s/step - loss: 0.0050 - accuracy: 0.9989 - val_loss: 0.2180 - val_accuracy: 0.9674
    Epoch 10/10
    115/115 [==============================] - 1310s 11s/step - loss: 0.0076 - accuracy: 0.9967 - val_loss: 0.1333 - val_accuracy: 0.9696

</div>

</div>

<div class="cell markdown" id="CIvA-4zI8YJa">

# Plot Accuracy and Loss

</div>

<div class="cell code" execution_count="17"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:747}"
id="Q7-WT4IV5lKH" outputId="044500ca-190a-47b8-f7ed-6b4396d7f403">

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

![](7a57ab0c2d47bcca89a35974e72fe289c652c4cb.png)

</div>

<div class="output display_data">

![](1cec8d0157cfa2ef44cea3d5063254782436030f.png)

</div>

</div>

<div class="cell markdown" id="j5G81-_u7q5i">

# **Predict test set**

</div>

<div class="cell code" execution_count="18"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="AHrDLyf65tdH" outputId="654b743e-6fcd-40a8-f6b1-f5495b8d771b">

``` python
model.evaluate(test, verbose=1)
```

<div class="output stream stdout">

    15/15 [==============================] - 35s 2s/step - loss: 0.0953 - accuracy: 0.9739

</div>

<div class="output execute_result" execution_count="18">

    [0.09531395882368088, 0.9739130139350891]

</div>

</div>

<div class="cell code" execution_count="19"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="sxn6liQ751kx" outputId="8eb5f4d3-26f6-4ef8-cae1-73b586093f36">

``` python
pred = model.predict(test)
y_pred = np.argmax(pred, axis=1)
```

<div class="output stream stdout">

    15/15 [==============================] - 37s 2s/step

</div>

</div>

<div class="cell code" execution_count="20" id="BCmqXZv852pa">

``` python
y_test = test.labels
y_test = np.array(y_test)
```

</div>

<div class="cell code" execution_count="21"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="cqY7Anaz53pN" outputId="cfa07119-85d4-4ce5-84ba-c1dca20494e7">

``` python
from sklearn.metrics import accuracy_score

print("Accuracy of the Model:",accuracy_score(y_test, y_pred)*100,"%")
```

<div class="output stream stdout">

    Accuracy of the Model: 97.3913043478261 %

</div>

</div>

<div class="cell code" execution_count="22"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:462}"
id="B-VAaLXm55Sj" outputId="451460de-009a-41b8-9369-37090adb078b">

``` python
from sklearn.metrics import confusion_matrix, accuracy_score

plt.figure(figsize = (10,5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt = 'g', cmap = 'crest')
```

<div class="output execute_result" execution_count="22">

    <Axes: >

</div>

<div class="output display_data">

![](3a17a651d423a416626c867d49b9777fac140a32.png)

</div>

</div>

<div class="cell markdown" id="RbAU6vCh8lK-">

# **Early Stopping**

</div>

<div class="cell markdown" id="cLiOINEQ9SZ4">

To save training time we can stop training the CNN if the accuracy of
the validation data does not improve after a certain number of steps.
For example, the selection criteria for early stopping is the accuracy
on the validation data, and the algorithm will stop training if the
validation accuracy does not improve by at least 0.5 after 5 epochs.

</div>

<div class="cell code" execution_count="25"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="ar5sgSHe8nSX" outputId="43d29fbe-f387-4819-9efd-b21400a23b56">

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

early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 5, min_delta=0.5)
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_2 = model_2.fit(train, epochs=10, verbose=1, validation_data = val, callbacks=[early_stop])
```

<div class="output stream stdout">

    Epoch 1/10
    115/115 [==============================] - 1325s 11s/step - loss: 4.9975 - accuracy: 0.7351 - val_loss: 0.2366 - val_accuracy: 0.9065
    Epoch 2/10
    115/115 [==============================] - 1324s 12s/step - loss: 0.1433 - accuracy: 0.9484 - val_loss: 0.1123 - val_accuracy: 0.9674
    Epoch 3/10
    115/115 [==============================] - 1323s 12s/step - loss: 0.0383 - accuracy: 0.9891 - val_loss: 0.1226 - val_accuracy: 0.9587
    Epoch 4/10
    115/115 [==============================] - 1323s 12s/step - loss: 0.0182 - accuracy: 0.9962 - val_loss: 0.1232 - val_accuracy: 0.9761
    Epoch 5/10
    115/115 [==============================] - 1317s 11s/step - loss: 0.0129 - accuracy: 0.9967 - val_loss: 0.1124 - val_accuracy: 0.9717
    Epoch 6/10
    115/115 [==============================] - 1319s 11s/step - loss: 0.0110 - accuracy: 0.9970 - val_loss: 0.0989 - val_accuracy: 0.9761

</div>

</div>

<div class="cell code" execution_count="26"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="cHxZxrwp91xi" outputId="f284659c-33cf-4730-d125-73d7609b3d79">

``` python
print('Train Accuracy with Early Stopping:', model_2.evaluate(test, verbose=1))
```

<div class="output stream stdout">

    15/15 [==============================] - 37s 2s/step - loss: 0.0951 - accuracy: 0.9696
    Train Accuracy with Early Stopping: [0.09507567435503006, 0.969565212726593]

</div>

</div>

<div class="cell code" execution_count="27"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:747}"
id="LwmJEe1PNf69" outputId="de428c03-c4a4-44da-c6c3-9306ddb18692">

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

![](68e395141f81730e67eef95905a5cc988bcd281c.png)

</div>

<div class="output display_data">

![](ea586a64c0d3aa05241259eb2135242eea6fb0a8.png)

</div>

</div>
