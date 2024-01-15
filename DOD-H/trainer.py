#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 15:50:40 2023

@author: alikavoosi
"""

import tensorflow as tf
from nn_model import *
from dreemRead import *
from scipy.signal import resample
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from scipy.signal import butter, lfilter

tf.random.set_seed(100)

# Function to create a bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply the bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


seq_len = 12

cnn_acc = []
seq_acc = []


import random

def get_fold_indices(fold_number, total_folds=25):
    
    if fold_number < 0 or fold_number >= total_folds:
        raise ValueError("Fold number is out of range")

    fold_size = total_folds // 25  # 25 is your total number of folds

    # Calculate the test index (1 number)
    test_index = fold_number

    # Calculate random validation indices (7 distinct integers)
    all_indices = list(range(total_folds))
    all_indices.remove(test_index)  # Remove test index

    validation_indices = random.sample(all_indices, 7)

    # Calculate the training indices (17 indices)
    training_indices = [
        i for i in range(total_folds) if i not in [test_index] + validation_indices
    ]

    return training_indices, [test_index], validation_indices

for fold in range(1,25):
    best_model_file = f'dreem_ase_model_fold{fold}.h5'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_file, 
                                                      monitor='val_loss', 
                                                      mode = 'min',
                                                      save_best_only=True,
                                                      save_freq="epoch")

    best_model_file_seq = f'dreem_seq_model_fold{fold}.h5'
    checkpoint_callback_seq = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_file_seq, 
                                                      monitor='val_loss', 
                                                      mode = 'min',
                                                      save_best_only=True,
                                                      save_freq="epoch")
    
    
    
    print('##################################################################')
    print(f'fold is {fold}')
    print('##################################################################')
    train_inds, test_inds, val_inds = get_fold_indices(fold)
    np.save(f'train_ind_dooh_fold{fold}.npy', train_inds)
    np.save(f'test_ind_dodh_fold{fold}.npy', test_inds)
    np.save(f'val_ind_dodh_fold{fold}.npy', val_inds)
    
    def convert_arr(x):
        l = 0
        for i in x:
            l += len(x)
        return np.reshape(np.array(x),(l,1,3000,1))
    
    
    x_train = []
    y_train = []
    for i in train_inds:
        x,hyp = extract_data(i)
        x = butter_bandpass_filter(x,0.5,40,250)
        x_r = resample(x, int(len(x)*100/250))
        inds = np.arange(0,len(x_r),30*100)
        epochs = np.reshape(x_r,(len(inds),1,3000,1))
        for num,i in enumerate(epochs):
            epochs[num]=(i-np.mean(i))/np.std(i)
        
        x_train.append(epochs)
        y_train.append(np.array(hyp).reshape((len(hyp),1)))
    
    x_train = np.vstack(x_train)
    y_train = np.vstack(y_train)
    
    x_val = []
    y_val = []
    for i in val_inds:
        x,hyp = extract_data(i)
        x = butter_bandpass_filter(x,0.5,40,250)
        x_r = resample(x, int(len(x)*100/250))
        inds = np.arange(0,len(x_r),30*100)
        epochs = np.reshape(x_r,(len(inds),1,3000,1))
        for num,i in enumerate(epochs):
            epochs[num]=(i-np.mean(i))/np.std(i)
        
        x_val.append(epochs)
        y_val.append(np.array(hyp).reshape((len(hyp),1)))
    
    x_val = np.vstack(x_val)
    y_val = np.vstack(y_val)
    
    x_test = []
    y_test = []
    for i in test_inds:
        x,hyp = extract_data(i)
        x = butter_bandpass_filter(x,0.5,40,250)
        x_r = resample(x, int(len(x)*100/250))
        inds = np.arange(0,len(x_r),30*100)
        epochs = np.reshape(x_r,(len(inds),1,3000,1))
        for num,i in enumerate(epochs):
            epochs[num]=(i-np.mean(i))/np.std(i)
        
        x_test.append(epochs)
        y_test.append(np.array(hyp).reshape((len(hyp),1)))
    
    x_test = np.vstack(x_test)
    y_test = np.vstack(y_test)
    
    x_train, y_train = shuffle(x_train, y_train)
    
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = 10e-3)
    model = separable_resnet((1,3000,1), 5, y_train = y_train, bias = False)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    model.fit(x_train, to_categorical(y_train), batch_size=32, epochs=5, 
              validation_data = (x_val, to_categorical(y_val)), callbacks = [checkpoint_callback])
    
    model = tf.keras.models.load_model(best_model_file)
    print('####################         Test Result       ##################')
    val,acc = model.evaluate(x_test, to_categorical(y_test))
    cnn_acc.append(acc)
    
    
    saved_model_dir = 'saveHere'
    # Save the model in SavedModel format
    tf.saved_model.save(model, saved_model_dir)
    
    x_rep = x_train[np.random.randint(0,len(x_train),500)]
    def representative_dataset():
        global x_rep
        for data in x_rep:
          yield [data.astype(np.float32).reshape((1,1,3000,1))]
    
    path = 'saveHere'
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_quant_model = converter.convert()
    
    tflite_path = f"cnn_full_int_fold_{fold}.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_quant_model)
        
        
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    output_details = interpreter.get_output_details()
    output_scale, output_zero_point = output_details[0]['quantization']
    
    input_details = interpreter.get_input_details()
    input_scale, input_zero_point = input_details[0]['quantization']
    
    
    
    
    x_train_seq = []
    y_train_seq = []
    
    x_val_seq = []
    y_val_seq = []
    
    x_test_seq = []
    y_test_seq = []
    
    
    for j in train_inds:
        x,hyp = extract_data(j)
        x = butter_bandpass_filter(x,0.5,40,250)
        x_r = resample(x, int(len(x)*100/250))
        inds = np.arange(0,len(x_r),30*100)
        epochs = np.reshape(x_r,(len(inds),1,3000,1))
        for num,i in enumerate(epochs):
            epochs[num]=(i-np.mean(i))/np.std(i)
        # pred = model.predict(epochs, verbose=0)
        
        # To use tflite: start
        pred = []
        for i in epochs:
            input_data = i.reshape((1,1,3000,1))
            # Get the input details (assuming a single input tensor)
            input_details = interpreter.get_input_details()
            input_index = input_details[0]["index"]
            input_scale, input_zero_point = input_details[0]['quantization']
            quantized_input = (input_data / input_scale) + input_zero_point
            quantized_input = quantized_input.astype(np.int8)
            
            # Set the input tensor
            interpreter.set_tensor(input_index, quantized_input)
            
            interpreter.invoke()
            
            # Get the output details (assuming a single output tensor)
            output_details = interpreter.get_output_details()
            output_index = output_details[0]["index"]
            
            # Get the output tensor
            output_data = interpreter.get_tensor(output_index)
            dequantized_output = (output_data.astype(np.float32) - output_zero_point) * output_scale
            pred.append(dequantized_output)
        # To ise tflite: end
        
        
        
        
        
        for i in range(seq_len,len(epochs)):
            # pred = model.predict(epochs[i-12:i], verbose=0)
            x_train_seq.append(pred[i-seq_len:i])
            y_train_seq.append(hyp[i-1])
        
    
    
    for j in val_inds:
        x,hyp = extract_data(j)
        x = butter_bandpass_filter(x,0.5,40,250)
        x_r = resample(x, int(len(x)*100/250))
        inds = np.arange(0,len(x_r),30*100)
        epochs = np.reshape(x_r,(len(inds),1,3000,1))
        for num,i in enumerate(epochs):
            epochs[num]=(i-np.mean(i))/np.std(i)
        # pred = model.predict(epochs, verbose=0)
        
        # To use tflite: start
        pred = []
        for i in epochs:
            input_data = i.reshape((1,1,3000,1))
            # Get the input details (assuming a single input tensor)
            input_details = interpreter.get_input_details()
            input_index = input_details[0]["index"]
            input_scale, input_zero_point = input_details[0]['quantization']
            quantized_input = (input_data / input_scale) + input_zero_point
            quantized_input = quantized_input.astype(np.int8)
            
            # Set the input tensor
            interpreter.set_tensor(input_index, quantized_input)
            
            interpreter.invoke()
            
            # Get the output details (assuming a single output tensor)
            output_details = interpreter.get_output_details()
            output_index = output_details[0]["index"]
            
            # Get the output tensor
            output_data = interpreter.get_tensor(output_index)
            dequantized_output = (output_data.astype(np.float32) - output_zero_point) * output_scale
            pred.append(dequantized_output)
        # To ise tflite: end
        
        for i in range(seq_len,len(epochs)):
            # pred = model.predict(epochs[i-12:i], verbose=0)
            x_val_seq.append(pred[i-seq_len:i])
            y_val_seq.append(hyp[i-1])
    
    
    for j in test_inds:
        x,hyp = extract_data(j)
        x = butter_bandpass_filter(x,0.5,40,250)
        x_r = resample(x, int(len(x)*100/250))
        inds = np.arange(0,len(x_r),30*100)
        epochs = np.reshape(x_r,(len(inds),1,3000,1))
        for num,i in enumerate(epochs):
            epochs[num]=(i-np.mean(i))/np.std(i)
        # pred = model.predict(epochs, verbose=0)
        
        # To use tflite: start
        pred = []
        for i in epochs:
            input_data = i.reshape((1,1,3000,1))
            # Get the input details (assuming a single input tensor)
            input_details = interpreter.get_input_details()
            input_index = input_details[0]["index"]
            input_scale, input_zero_point = input_details[0]['quantization']
            quantized_input = (input_data / input_scale) + input_zero_point
            quantized_input = quantized_input.astype(np.int8)
            
            # Set the input tensor
            interpreter.set_tensor(input_index, quantized_input)
            
            interpreter.invoke()
            
            # Get the output details (assuming a single output tensor)
            output_details = interpreter.get_output_details()
            output_index = output_details[0]["index"]
            
            # Get the output tensor
            output_data = interpreter.get_tensor(output_index)
            dequantized_output = (output_data.astype(np.float32) - output_zero_point) * output_scale
            pred.append(dequantized_output)
        # To ise tflite: end
        
        for i in range(seq_len,len(epochs)):
            # pred = model.predict(epochs[i-12:i], verbose=0)
            x_test_seq.append(pred[i-seq_len:i])
            y_test_seq.append(hyp[i-1])
    
    x_train_seq = np.reshape(np.array(x_train_seq),(len(x_train_seq),int(seq_len*5)))
    x_val_seq = np.reshape(np.array(x_val_seq),(len(x_val_seq),int(seq_len*5)))
    x_test_seq = np.reshape(np.array(x_test_seq),(len(x_test_seq),int(seq_len*5)))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = 10e-3)
    seq_model = seq_model(int(seq_len*5))
    seq_model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    seq_model.fit(x_train_seq, to_categorical(y_train_seq), batch_size=32, epochs=15, 
              validation_data = (x_val_seq, to_categorical(y_val_seq)), callbacks = [checkpoint_callback_seq])
    seq_model = tf.keras.models.load_model(best_model_file_seq)
    print('####################         Test Result       ##################')
    val,acc = seq_model.evaluate(x_test_seq, to_categorical(y_test_seq))
    seq_acc.append(acc)
