import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_recall_fscore_support as score

import sys
sys.path.append('../')
from global_methods import *
from global_variables import *

def get_F1_score(label, prediction):
  TP = 0
  TN = 0
  FP = 0
  FN = 0

  for i in range(len(label)): 
    if label[i] == 1 and prediction[i] == 1: TP += 1
    if label[i] == 0 and prediction[i] == 0: TN += 1
    if label[i] == 0 and prediction[i] == 1: FP += 1
    if label[i] == 1 and prediction[i] == 0: FN += 1

  accuracy = float(TP + TN)/len(label) * 100
  precision = float(TP)/(TP + FP) * 100
  recall = float(TP)/(TP + FN) * 100

  return (2 * precision*recall) / (precision + recall)


def get_word_index(training_data, vocab_size): 
  word_count = dict()
  for count, row in enumerate(training_data):
    if row: 
      all_row_words = row.split()
      for word in all_row_words:
        if word in word_count: 
          word_count[word] += 1
        else: 
          word_count[word] = 1

  vocab_size -= 4
  word_count_flattened =[(val, key) for (key, val) in word_count.items()][:vocab_size]
  word_count_flattened.sort(reverse=True)

  word_index = dict()
  word_index["<PAD>"] = 0
  word_index["<START>"] = 1
  word_index["<UNK>"] = 2
  word_index["<UNUSED>"] = 3

  for count, i in enumerate(word_count_flattened): 
    word_index[i[1]] = count + 4

  return word_index


def encode(curr_str, word_index): 
  # Starting with 1 as that marks the start of the text doc.
  encoded_str = [word_index["<START>"]]
  curr_str = curr_str.split()
  for word in curr_str:
    if word.lower() in word_index:
      encoded_str.append(word_index[word.lower()])
    else: 
      encoded_str.append(word_index["<UNK>"])
  return encoded_str


def load_all_data(all_content, training_count): 
  data = []
  label = []
  for count in range(len(all_content)): 
    data += [all_content[count][2]]
    if all_content[count][1] == "removed": 
      label += [1]
    else: 
      label += [0]

  mixed_data = []
  mixed_label = []

  stop_sign = int(len(data)/2)
  for count in range(len(data)): 
    if stop_sign == count: break
    mixed_data += [data[count], data[count*-1 - 1]]
    mixed_label += [label[count], label[count*-1 - 1]]

  training_data = mixed_data[:training_count]
  training_label = mixed_label[:training_count]

  testing_data = mixed_data[training_count:]
  testing_label = mixed_label[training_count:]

  return (training_data, training_label), (testing_data, testing_label)


def train(vocab_size, 
          maxlen, 
          training_data_count, 
          validation_data_count, 
          dense_layer_node,
          epochs, 
          all_comments):
  (training_data, training_label), (testing_data, testing_label) = load_all_data(all_comments, training_data_count)

  word_index = get_word_index(training_data, vocab_size)
  reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

  print ("FINISHED LOADING")

  encoded_training_data = [encode(i, word_index) for i in training_data]
  encoded_testing_data = [encode(i, word_index) for i in testing_data]

  encoded_training_data = keras.preprocessing.sequence.pad_sequences(encoded_training_data, value=word_index["<PAD>"], padding="post", maxlen=maxlen)
  encoded_testing_data = keras.preprocessing.sequence.pad_sequences(encoded_testing_data, value=word_index["<PAD>"], padding="post", maxlen=maxlen)

  print ("FINISHED ENCODING")
  print (len(word_index))

  model = keras.Sequential()
  model.add(keras.layers.Embedding(vocab_size, dense_layer_node))
  model.add(keras.layers.GlobalAveragePooling1D())
  model.add(keras.layers.Dense(dense_layer_node, activation=tf.nn.relu))
  model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

  model.summary()
  model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

  x_val = encoded_training_data[:validation_data_count]
  x_train = encoded_training_data[validation_data_count:]

  y_val = training_label[:validation_data_count]
  y_train = training_label[validation_data_count:]

  fitModel = model.fit(x_train, y_train, epochs=epochs, batch_size=512, validation_data=(x_val, y_val), verbose=1)


  results = model.evaluate(encoded_testing_data, testing_label)
  print (results)

  prediction = model.predict(encoded_testing_data)
  output = []
  for i in prediction: 
    if i[0] >= 0.5:
      output += [1]
    else: 
      output += [0]
  f1score = get_F1_score(testing_label, output)
  print (f1score)
  
  return (model, results, f1score)


def __Full_Basic_Train_Routine(subreddit, verbose_write): 
  # Open data
  data_location = "clean_data/" + subreddit + ".csv"
  all_comments = read_file_to_list(data_location)[1:]

  # Size of the embedding
  vocab_size = 88000
  # Max length of the document
  maxlen = 256
  # The amount of training (testing) data + validation subset of the training data
  training_data_count = int(len(all_comments) * 0.7) 
  validation_data_count = int(training_data_count * 0.3)
  # The number of epochs to run
  epochs = 20
  dense_layer_node = 16 #16

  verbose_write = verbose_write

  # Model
  model, results, f1score = train(vocab_size, 
                                  maxlen, 
                                  training_data_count, 
                                  validation_data_count, 
                                  dense_layer_node,
                                  epochs, 
                                  all_comments)

  if verbose_write: 
    model_outfile = "all_nn_models/" + subreddit + ".h5"
    model.save(model_outfile)

    stat_outfile = "all_nn_models_stats/" + subreddit + ".csv"
    stat_out = [["vocab_size", vocab_size],
                ["maxlen", maxlen],
                ["training_data_count", training_data_count],
                ["validation_data_count", validation_data_count],
                ["dense_layer_node", dense_layer_node],
                ["epochs", epochs],
                ["accuracy", results[1]],  
                ["F1 Score", f1score]]
    write_list_of_list_to_csv(stat_out, stat_outfile)


def __Full_Hyper_Train_Routine(subreddit): 
  # Open data
  data_location = "clean_data/" + subreddit + ".csv"
  all_comments = read_file_to_list(data_location)[1:]

  # Size of the embedding
  vocab_size = [10000, 44000]
  # Max length of the document
  maxlen = [256, 512]
  # The amount of training (testing) data + validation subset of the training data
  # This rate will give 70 for training, 15 for testing and validation
  training_data_count = int(len(all_comments) * 0.85) 
  validation_data_count = int(training_data_count * 0.17647)
  # The number of epochs to run
  epochs = [30, 40, 50]
  dense_layer_node = [16, 32] #16

  progress_count = 0
  for vs in vocab_size: 
    for ml in maxlen: 
      for ep in epochs: 
        for dln in dense_layer_node: 
          progress_count += 1
          print ("PROGRESS COUNT ::: ", subreddit, " ::: ", progress_count)
          print ("Curr parameters ::: vs, ml dln, ep :::", vs, ml, dln, ep)

          model, results, f1score = train(vs, 
                                          ml, 
                                          training_data_count, 
                                          validation_data_count, 
                                          dln,
                                          ep, 
                                          all_comments)

          try:
            existing_f1score = float(read_file_to_list("all_nn_models_stats/" + subreddit + ".csv")[-1][-1])
          except: 
            existing_f1score = -1

          # go into verbose_mode if F1 score is higher than what we have (or if we dno't have one already)
          if f1score > existing_f1score: 
            print ("NEW BEST PARAMETER ----- ", subreddit)

            model_outfile = "all_nn_models/" + subreddit + ".h5"
            model.save(model_outfile)

            stat_outfile = "all_nn_models_stats/" + subreddit + ".csv"
            stat_out = [["vocab_size", vs],
                        ["maxlen", ml],
                        ["training_data_count", training_data_count],
                        ["validation_data_count", validation_data_count],
                        ["dense_layer_node", dln],
                        ["epochs", ep],
                        ["accuracy", results[1]],  
                        ["F1 Score", f1score]]
            write_list_of_list_to_csv(stat_out, stat_outfile)
