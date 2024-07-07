import keras.datasets as datasets
import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split

# CIFAR100 images
(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

selected_classes = [69, 85, 90]

selected_indices_train = np.where(np.isin(y_train, selected_classes))[0]

x_train = x_train[selected_indices_train]
y_train = y_train[selected_indices_train]

selected_indices_test = np.where(np.isin(y_test, selected_classes))[0]

x_test = x_test[selected_indices_test]
y_test = y_test[selected_indices_test]

class_mapping = {69: 0, 85: 1, 90: 2}

y_train = np.array([class_mapping[val] for val in y_train.flatten()])

y_test = np.array([class_mapping[val] for val in y_test.flatten()])

# CIFAR10 images

(x_train_2, y_train_2), (x_test_2, y_test_2) = datasets.cifar10.load_data()

selected_classes_2 = [0, 8]

selected_indices_train_2 = np.where(np.isin(y_train_2, selected_classes_2))[0]

x_train_2 = x_train_2[selected_indices_train_2]
y_train_2 = y_train_2[selected_indices_train_2]

selected_indices_test_2 = np.where(np.isin(y_test_2, selected_classes_2))[0]

x_test_2 = x_test_2[selected_indices_test_2]
y_test_2 = y_test_2[selected_indices_test_2]

class_mapping_2 = {0: 3, 8:4}

y_train_2 = np.array([class_mapping_2[val] for val in y_train_2.flatten()])

y_test_2 = np.array([class_mapping_2[val] for val in y_test_2.flatten()])

# Random selecting class images from CIFAR10 so that their amount matches with CIFAR100 classes

selected_3_indices_train = random.sample([i for i, x in enumerate(y_train_2) if x == 3], 500)
selected_4_indices_train = random.sample([i for i, x in enumerate(y_train_2) if x == 4], 500)

x_train_2 = np.concatenate((x_train_2[selected_3_indices_train],x_train_2[selected_4_indices_train]))
y_train_2 = np.append(y_train_2[selected_3_indices_train],y_train_2[selected_4_indices_train])

selected_3_indices_test = random.sample([i for i, x in enumerate(y_test_2) if x == 3], 100)
selected_4_indices_test = random.sample([i for i, x in enumerate(y_test_2) if x == 4], 100)

x_test_2 = np.concatenate((x_test_2[selected_3_indices_test],x_test_2[selected_4_indices_test]))
y_test_2 = np.append(y_test_2[selected_3_indices_test],y_test_2[selected_4_indices_test])

# Combining CIFAR10 and CIFAR100 images

x_train = np.concatenate((x_train, x_train_2))
y_train = np.concatenate((y_train, y_train_2))

x_test = np.concatenate((x_test, x_test_2))
y_test = np.concatenate((y_test, y_test_2))

# Train validation split

x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=500, random_state=42)

x_train = x_train[:2000]
y_train = y_train[:2000]

x_validation = x_validation[:500]
y_validation = y_validation[:500]

pure_dataset = {'x_test' : x_test, 'y_test' :y_test, 'x_validation' :x_validation,
                'y_validation' : y_validation, 'x_train' :x_train, 'y_train' :y_train}

with open('Dataset/dataset.pickle', 'wb') as f:
    pickle.dump(pure_dataset, f)

# Preparing dataset for model

x_train = x_train.astype('float64')/ 255.0
x_validation = x_validation.astype('float64')/ 255.0
x_test = x_test.astype('float64') /255.0

mean = np.mean(x_train, axis = 0)
std = np.std(x_train, axis=0)

mean_std = {'mean': mean, 'std': std}

with open('Model/image_preparation.pickle', 'wb') as f:
    pickle.dump(mean_std,f)

x_train = (x_train - mean)/std
x_validation = (x_validation - mean)/std
x_test = (x_test - mean)/std

x_train = x_train.transpose(0, 3, 1, 2)
x_test = x_test.transpose(0, 3, 1, 2)
x_validation = x_validation.transpose(0, 3, 1, 2)

model_prepared_dataset = {'x_test' : x_test, 'y_test' :y_test, 'x_validation' :x_validation,
                          'y_validation' : y_validation, 'x_train' :x_train, 'y_train' :y_train}

with open('Dataset/dataset_for_model.pickle', 'wb') as f:
    pickle.dump(model_prepared_dataset, f)

