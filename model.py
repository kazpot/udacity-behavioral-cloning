import matplotlib.image as img
import numpy as np
import pandas as pd
import csv
import json
from keras.preprocessing.image import ImageDataGenerator, Iterator, flip_axis
from keras.layers.core import Dense, Flatten, Activation, SpatialDropout2D, Lambda, Dropout
from keras.models import Model, Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data_path = '/home/kazuhiro/DrivingData_UDACITY/driving_log.csv'
#data_path = '/home/kazuhiro/DrivingData2/driving_log.csv'
#data_path = '/home/kazuhiro/DrivingData/driving_log.csv'

#tuning parameter
angle_correction = 0.07
flip_prob = 0.5
batch_size = 128
nb_epoch = 5


class MyDataGenerator(ImageDataGenerator):
    def flow(self, X, y=None, batch_size=32, shuffle=False, seed=None, flip_prob=0):
        return MyIterator(X, y, batch_size=batch_size, shuffle=shuffle, seed=seed, flip_prob=flip_prob)


class MyIterator(Iterator):
    def __init__(self, X, y, batch_size=32, shuffle=False, seed=None, flip_prob=0):

        self.X = X
        self.y = y
        self.flip_prob = flip_prob

        super(MyIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x, batch_y = None, None
        for batch_idx, source_idx in enumerate(index_array):
            x = img.imread(self.X[source_idx])
            y = self.y[source_idx]

            if batch_x is None:
                batch_x = np.zeros(tuple([current_batch_size] + list(x.shape)))
                batch_y = np.zeros(current_batch_size)

            if np.random.choice([True, False], p=[self.flip_prob, 1.0 - self.flip_prob]):
                x = flip_axis(x, 1)
                y *= -1

            batch_x[batch_idx] = x
            batch_y[batch_idx] = y
            
        return batch_x, batch_y

def resize(X):
    import tensorflow
    return tensorflow.image.resize_images(X, (40, 160))


def run():
    
    driving_log = pd.read_csv(data_path, names=('Center Image', 'Left Image', 'Right Image', 'Steering Angle', 'Throttle', 'Break', 'Speed'))

    image_names_full=[] 
    y_data_full = []

    for index, row in driving_log.iterrows():
        center_img = row['Center Image']
        left_img = row['Left Image'].strip()
        right_img = row['Right Image'].strip()
        steering_angle = row['Steering Angle']

        image_names_full.append(center_img)
        y_data_full.append(steering_angle)

        left = steering_angle + angle_correction
        right = steering_angle - angle_correction

        image_names_full.append(left_img)
        y_data_full.append(left)

        image_names_full.append(right_img)
        y_data_full.append(right)
    image_names_full, y_data_full = np.array(image_names_full), np.array(y_data_full)
    
    print('CSV loaded')

    #split data
    X_train, X_val, y_train, y_val = train_test_split(image_names_full, y_data_full, test_size=0.2)

    #model
    model = Sequential()
    model.add(Cropping2D(cropping=((60, 20), (0, 0)),input_shape=(160,320,3)))
    model.add(Lambda(resize))
    model.add(BatchNormalization(axis=1))
    
    model.add(Convolution2D(24, 5, 5, border_mode='same', activation='elu'))
    model.add(MaxPooling2D(border_mode='same'))
    model.add(SpatialDropout2D(0.2))

    model.add(Convolution2D(36, 5, 5, border_mode='same', activation='elu'))
    model.add(MaxPooling2D(border_mode='same'))
    model.add(SpatialDropout2D(0.2))

    model.add(Convolution2D(48, 5, 5, border_mode='same', activation='elu'))
    model.add(MaxPooling2D(border_mode='same'))
    model.add(SpatialDropout2D(0.2))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='elu'))
    model.add(MaxPooling2D(border_mode='same'))
    model.add(SpatialDropout2D(0.2))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='elu'))
    model.add(MaxPooling2D(border_mode='same'))
    model.add(SpatialDropout2D(0.2))
    
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(100, activation='elu',W_regularizer=l2(1e-6)))
    model.add(Dense(50, activation='elu',W_regularizer=l2(1e-6)))
    model.add(Dense(10, activation='elu',W_regularizer=l2(1e-6)))
    model.add(Dense(1))

    #summary
    model.summary()

    #training
    print('Start training')
    model.compile(optimizer='adam', loss='mse')
    datagen = MyDataGenerator()
    history = model.fit_generator(
            datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True, flip_prob=flip_prob),
            samples_per_epoch=len(y_train),
            nb_epoch=nb_epoch,
            validation_data=datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=True),
            nb_val_samples=len(y_val))

    #save model
    print('Save model')
    with open('model.json', 'w') as f:
        json.dump(model.to_json(), f)
    model.save_weights('model.h5')


if __name__ == "__main__":
    run()
