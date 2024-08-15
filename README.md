# 2.Solving-XOR-problem-using-deep-feed-forward-network.
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
model=Sequential()
model.add(Dense(units=2,activation='relu',input_dim=2))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
print(model.get_weights())
X=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y=np.array([0.,1.,1.,0.])
model.fit(X,Y,epochs=1000,batch_size=4)
print(model.get_weights())
print(model.predict(X,batch_size=4))


OR 

import numpy as np
from keras.layers import Dense
from keras.models import Sequential

# Initialize the model
model = Sequential()

# Add the first Dense layer
model.add(Dense(units=2, activation="relu", input_dim=2))

# Add the second Dense layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
print(model.summary())

# Print the model's initial weights (before training)
print("Initial weights:", model.get_weights())

# Define the training data
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = np.array([0., 1., 1., 0.])

# Train the model
model.fit(X, Y, epochs=1000, batch_size=4)

# Print the model's final weights (after training)
print("Final weights:", model.get_weights())

# Make predictions using the trained model
predictions = model.predict(X, batch_size=4)
print("Predictions:", predictions)

