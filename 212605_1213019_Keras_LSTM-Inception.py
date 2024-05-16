get_ipython().run_line_magic('run', 'Setup.ipynb')
get_ipython().run_line_magic('run', 'ExtraFunctions.ipynb')

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

l_lstm1 = Bidirectional(LSTM(4,dropout=0.3,recurrent_dropout=0.3,return_sequences=True))(embedded_sequences)

# inception module
inception, filter_sizes = [], [3,4,6,7]
for fsz in filter_sizes:
    l_conv_i1 = Conv1D(filters=12,kernel_size=1,
                    activation='relu',)(l_lstm1)
    l_conv_i2 = Conv1D(filters=12,kernel_size=fsz,
                       activation='relu',kernel_regularizer=regularizers.l2(0.01))(l_conv_i1)
    inception.append(l_conv_i2)
l_pool_inc = MaxPooling1D(4)(l_lstm1)
l_conv_inc = Conv1D(filters=12,kernel_size=1,
                    activation='relu',kernel_regularizer=regularizers.l2(0.02))(l_pool_inc)
inception.append(l_conv_inc)

l_merge = Concatenate(axis=1)(inception)
l_pool1 = MaxPooling1D(3)(l_merge)
l_drop1 = Dropout(0.4)(l_pool1)
l_flat = Flatten()(l_drop1)
l_dense = Dense(16, activation='relu',kernel_regularizer=regularizers.l2(0.02))(l_flat)

preds = Dense(4, activation='softmax')(l_dense)

model = Model(sequence_input, preds)
adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
lr_metric = get_lr_metric(adadelta)
#keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=adadelta,
              metrics=['acc', lr_metric])

tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=16, write_grads=True , write_graph=True)
model_checkpoints = callbacks.ModelCheckpoint("checkpoint.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
lr_schedule = callbacks.LearningRateScheduler(initial_boost)

model.summary()
model.save('ltsm-inception.h5')
print("Training Progress:")
model_log = model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=200, batch_size=50,
          callbacks=[tensorboard, lr_schedule])

model.save('ltsm-inception.h5')

import pandas as pd
pd.DataFrame(model_log.history).to_csv("history-inception.csv")

