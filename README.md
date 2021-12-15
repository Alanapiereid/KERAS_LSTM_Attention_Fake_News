# KERAS_LSTM_Attention_Fake_News
Fake News detection with Keras model using Kaggle Fake News dataset


____________________________________________________________________________________________________________________________________________________________



Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 300, 100)          5000000
_________________________________________________________________
lstm (LSTM)                  (None, 10)                4440
_________________________________________________________________
dense (Dense)                (None, 1)                 11
=================================================================
Total params: 5,004,451
Trainable params: 5,004,451
Non-trainable params: 0
_________________________________________________________________
2021-12-15 13:07:33.449676: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
Epoch 1/2
86/86 [==============================] - 30s 321ms/step - loss: 0.5318 - accuracy: 0.7628 - val_loss: 0.3628 - val_accuracy: 0.8895
Epoch 2/2
86/86 [==============================] - 26s 305ms/step - loss: 0.2585 - accuracy: 0.9442 - val_loss: 0.2428 - val_accuracy: 0.9304
Results - Loss: 0.2309851497411728 - Accuracy: 93.30708384513855%


