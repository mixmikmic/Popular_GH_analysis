get_ipython().run_line_magic('run', 'Setup.ipynb')

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

l_lstm1 = Bidirectional(LSTM(4,dropout=0.3,recurrent_dropout=0.3,return_sequences=True))(embedded_sequences)

convs, filter_sizes = [], [3,4,5]
for fsz in filter_sizes:
    l_conv = Conv1D(filters=32,kernel_size=fsz,
                    activation='relu',kernel_regularizer=regularizers.l2(0.01))(l_lstm1)
    convs.append(l_conv)

l_merge = Concatenate(axis=1)(convs)
l_pool1 = MaxPooling1D(2)(l_merge)
l_drop1 = Dropout(0.4)(l_pool1)
l_flat = Flatten()(l_drop1)
l_dense = Dense(16, activation='relu')(l_flat)

preds = Dense(4, activation='softmax')(l_dense)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

model = Model(sequence_input, preds)
adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
lr_metric = get_lr_metric(adadelta)
model.compile(loss='categorical_crossentropy',
              optimizer=adadelta,
              metrics=['acc', lr_metric])

def step_cyclic(epoch):
    try:
        l_r, decay = 1.0, 0.0001
        if epoch%33==0:multiplier = 10
        else:multiplier = 1
        rate = float(multiplier * l_r * 1/(1 + decay * epoch))
        #print("Epoch",epoch+1,"- learning_rate",rate)
        return rate
    except Exception as e:
        print("Error in lr_schedule:",str(e))
        return float(1.0)
    
def initial_boost(epoch):
    if epoch==0: return float(6.0)
    else: return float(1.0)
        
tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=4, batch_size=16, write_grads=True , write_graph=True)
model_checkpoints = callbacks.ModelCheckpoint("checkpoint.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
lr_schedule = callbacks.LearningRateScheduler(step_cyclic)

model.summary()
print("Training Progress:")
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=200, batch_size=50,
          callbacks=[tensorboard, model_checkpoints, lr_schedule])

model.save('ltsm-c.h5')



