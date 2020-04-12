import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
import data_processing

data = data_processing.load_data(download=False)
new_data = data_processing.convert2onehot(data)


# prepare training data
new_data = new_data.values.astype(np.float32)       # change to numpy array and float32
np.random.shuffle(new_data)
sep = int(0.7*len(new_data))
X_train = new_data[:sep, :21]                        # training data (70%)
y_train = new_data[:sep, 21:]
X_test = new_data[sep:, :21]                        # testing data (30%)
y_test = new_data[sep:, 21:]



# build network
model = Sequential()
# hidden layer
model.add(Dense(units=128, input_dim=21))
model.add(Activation('relu'))
model.add(Dense(units=128))
model.add(Activation('relu'))
# output layer
model.add(Dense(units=4))
model.add(Activation('softmax'))

# optimizer
adam = Adam(0.01)

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

BATCH_INDEX = 0
BATCH_SIZE = 32
# training
for step in range(1,2001):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    loss = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    if step % 50 == 0:
        loss, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print("test loss: %.6f" % loss, "  |  test accuracy: %.4f" % accuracy)