from keras.callbacks import ModelCheckpoint, EarlyStopping
from model import load_model
import numpy as np

# loading the training data
X_train = np.load('UCSD_training.npy')
frames = X_train.shape[2]

# Need to make number of frames divisible by 10
frames = frames-frames % 10
# reshaping the data
X_train = X_train[:, :, :frames]
X_train = X_train.reshape(-1, 227, 227, 10)
X_train = np.expand_dims(X_train, axis=4)
Y_train = X_train.copy()

# defining number of epochs and batch size
epochs = 32
batch_size = 1

if __name__ == "__main__":

    model = load_model()

    callback_save = ModelCheckpoint("Anomaly_Detector_UCSD.h5",
                                    monitor="mean_squared_error", save_best_only=True)

    callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    print('Model has been loaded')

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[callback_save, callback_early_stopping]
              )
    model.save('Anomaly_detector_UCSD.h5')
    print('model saved')
