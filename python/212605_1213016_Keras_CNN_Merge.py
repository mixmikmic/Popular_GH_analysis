import plaidml.keras
plaidml.keras.install_backend()

get_ipython().run_line_magic('run', 'Setup.ipynb')
get_ipython().run_line_magic('run', 'ExtraFunctions.ipynb')

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

convs, filter_sizes = [], [2,3,5]
for fsz in filter_sizes:
    l_conv = Conv1D(filters=16,kernel_size=fsz,activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(2)(l_conv)
    convs.append(l_pool)

l_merge = Concatenate(axis=1)(convs)
l_cov1= Conv1D(24, 3, activation='relu')(l_merge)
l_pool1 = MaxPooling1D(2)(l_cov1)
l_drop1 = Dropout(0.4)(l_pool1)
l_cov2 = Conv1D(16, 3, activation='relu')(l_drop1)
l_pool2 = MaxPooling1D(17)(l_cov2) # global max pooling
l_flat = Flatten()(l_pool2)
l_dense = Dense(16, activation='relu')(l_flat)
preds = Dense(6, activation='softmax')(l_dense)

model = Model(sequence_input, preds)
adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer="rmsprop",
              metrics=['accuracy'])

#tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=16, write_grads=True , write_graph=True)
model_checkpoints = callbacks.ModelCheckpoint("checkpoints", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
lr_schedule = callbacks.LearningRateScheduler(initial_boost)

get_ipython().system('rm -R logs')

model.summary()
model.save("cnn_kim.h5")

print("Training Progress:")
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=50)



