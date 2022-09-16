#%%
import tensorflow as tf
import os.path
import pickle
import csv
import os
import re
import string
from superdebug import debug
from tensorflow import keras
from sklearn.metrics import precision_recall_fscore_support as score
from tqdm import tqdm
import sys
sys.path.append('tools/subreddit_classifications/src/classifier_lib/')
from global_methods import *
from global_variables import *
from nn_train import *


def _text_preprocessing_helper(text): 
  """
  Takes in a string, and cleans up in various ways (e.g. remove special 
  characters, punctuations, etc.)
  ARGS:
    text: input text
  RETURNS: 
    text: cleaned up output text
  """
  ### 1) Remove newline and make it lowercase
  text = text.replace('\n', ' ').lower()
  ### 2) Make text lowercase, remove text in square brackets, remove 
  ###    punctuation and remove words containing numbers
  text = re.sub('\[.*?\]', '', text)
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub('\w*\d\w*', '', text)
  ### 3) Get rid of some additional punctuation and non-sensical text that was
  ###    missed the first time around
  text = re.sub('[‘’“”…]', '', text)
  ### 4) Strip whitespaces
  text = text.strip()
  text = ' '.join(text.split())

  return text


def load_word_index(subreddits): 
  """
  Loads all word indices of the classifiers that we are using into a 
  dictionary. 
  
  ARGS: 
    subreddits: a list of subreddits whose classifiers you want to use. 
  RETURNS: 
    all_word_index: a dictionary whose key is the name of the subreddit you are
                    using, and val is its associated word index. 
  """
  all_word_index = dict()

  for count, subreddit in enumerate(subreddits):
    nn_model_stats_file = "tools/subreddit_classifications/src/classifier_lib/all_nn_models_stats/" + subreddit + ".csv"
    nn_model_stats = read_file_to_list(nn_model_stats_file)
    vocab_size = int(nn_model_stats[0][1])
    training_data_count = int(nn_model_stats[2][1])

    curr_index_file = "tools/subreddit_classifications/src/classifier_lib/word_index/" + subreddit + "__VS" + str(vocab_size) + "__TDC" + str(training_data_count) + ".pkl"
    
    if os.path.isfile(curr_index_file): 
      # LOAD WORD INDEX
      pickle_in = open(curr_index_file, "rb")
      curr_index = pickle.load(pickle_in)
      all_word_index[subreddit] = curr_index
      pickle_in.close() 

    else: 
      # IF THE WORD INDEX DOES NOT EXIST ALREADY, CREATE ONE
      data_location = "tools/subreddit_classifications/src/classifier_lib/clean_data/" + subreddit + ".csv"
      all_comments = read_file_to_list(data_location)[1:]
      (training_data, training_label), (testing_data, testing_label) = load_all_data(
                                                                 all_comments,
                                                                 training_data_count)
      curr_index = get_word_index(training_data, vocab_size)

      with open(curr_index_file, "wb") as f:  
        pickle.dump(curr_index, f)
        all_word_index[subreddit] = curr_index

      print ("New word index pickled at: ", curr_index_file)

  return all_word_index


def analyze_all_comments_with_one_model(model_subreddit, 
                                        subreddit_input, 
                                        word_index): 
  """
  Conducts classifications of all comments using one subreddit classifier. 
  ARGS: 
    model_subreddit: the name of the subreddit whose classifier you want to
                     use. 
    subreddit_input: the list of text strings that you want to classify. 
    word_index: the word index associated with the subreddit whose classifier
                you want to use. 
  RETURNS: 
    final_analysis: list of list where each row is the result of the 
                    classifications for each comments. The order of the
                    subreddit_input is preserved. 
  """
  # Load in the hyperparameters for this subreddit classifer. 
  nn_model_stats_file = "tools/subreddit_classifications/src/classifier_lib/all_nn_models_stats/" + model_subreddit + ".csv"
  nn_model_stats = read_file_to_list(nn_model_stats_file)
  maxlen = int(nn_model_stats[1][1])

  # Load the subreddit classifier. 
  model_location = "tools/subreddit_classifications/src/classifier_lib/all_nn_models/" + model_subreddit + ".h5"
  model = keras.models.load_model(model_location)

  outlist = []
  # print ("Classifying ", len(subreddit_input), "rows...")
  for count, test_str in enumerate(subreddit_input): 
    # if count % 5000 == 0: print (count, len(subreddit_input)) # DEBUG
    # Encode the text and preprocess (pad). 
    encoded = encode(test_str, word_index)
    encoded = keras.preprocessing.sequence.pad_sequences([encoded], 
                                                     value=word_index["<PAD>"],
                                                     padding="post", 
                                                     maxlen=maxlen)
    # Conduct the actual classification of the comment and record. 
    outlist += [[test_str, model.predict(encoded)[0][0]]]

  return outlist


def run_all_classification(subreddit_input, subreddits_model_list): 
  """
  Takes in a list of str that we want to classify using our subreddit 
  classifiers (e.g. list of subreddit comments) and conducts classifications.
  It is worth noting that loading a subreddit classifier takes time, so instead
  of feeding in one comment at a time, it is more efficient to load a list of 
  many comments at once to classify. 
  There are 97 subreddits classifiers in total. You can use all of them or only
  use a few of them. The subreddits_model_list argument should be the list of 
  all subreddit names whose classifiers you want to use. 
  ARGS: 
    subreddit_input: a list of strings that you want to classify. 
    subreddits_model_list: a list of subreddits whose classifiers you want to 
                          use. 
  RETURNS: 
    final_analysis: list of list where each row is the result of the 
                    classifications for each comments. The order of the
                    subreddit_input is preserved. 
  """
  # The classifiers are made with neural net. The first step of using it is to 
  # load the word index. All subreddit classifiers have a word index that is 
  # unique to them. 
  all_word_index = load_word_index(subreddits_model_list)

  # Doing the actual classification. We condut all classifications with one 
  # subreddit classifier, and then move on to the next one, and so on. 
  all_subreddit_model_outputs = []
  for count, model_subreddit in enumerate(tqdm(subreddits_model_list)): 
    # print ("processing with ", model_subreddit, " model...", count)
    outlist = analyze_all_comments_with_one_model(model_subreddit, 
                                              subreddit_input, 
                                              all_word_index[model_subreddit])
    # analyze_all_comments_with_one_model will get you a list of floats (and 
    # the original comment in index 0). 
    # Each corresponds to the classification output for each of the comments. 
    # To make this into a binary classification, we set the threshold at 0.5
    # and determine a comment to be violating if its classification outcome is
    # greater than 0.5. 
    # This binary-fication happens here. 
    outlist = [i[1] for i in outlist]
    f_outlist = []
    for i in outlist:
      if i >= 0.5: 
        f_outlist += [1]
      else: 
        f_outlist += [0]
    # all_subreddit_model_outputs is ultimately a list of list whose rows 
    # indicate the binary ouputs for every comemnts using one subreddit
    # classifier. 
    all_subreddit_model_outputs += [f_outlist]

  # transpose
  final_analysis = [[] for i in subreddit_input]
  for count_i, model in enumerate(all_subreddit_model_outputs): 
    for count_j, comment in enumerate(subreddit_input): 
      final_analysis[count_j] += [model[count_j]]

  return final_analysis
#%%
def get_classifier_agreement_scores(contents):
  contents = [_text_preprocessing_helper(content) for content in contents]
  models_analysis = run_all_classification(contents, ALL_ACTIVE_SUBREDDITS)
  assert len(contents) == len(models_analysis)
  classifier_agreement_scores = [sum(_) for _ in models_analysis]
  return classifier_agreement_scores
#%%
def output_classification(in_file, comment_body_col, out_file):
  """
  Takes in an input csv file that contains all the comment rows you want to 
  classifiy. Conducts classifications using all subreddit classifiers and 
  outputs the result to the directory designated as out_file. 
  The outputted csv file is structured so that the subreddit classifications
  are appended directly following the original rows. The classification 
  outputs designate whether the content in question is problematic according 
  to the subreddit classifiers (1 if it is, 0 if it is not). 
  ARGS:
    in_file: The directory for the input file. The file needs to be a csv file
             and contains a header. One of the columns must be the text body 
             that you want to classify. 
    comment_body_col: Designates the column that contains the text body in the
                      input file. 
    out_file: The directory for the output file. 
  RETURNS: 
    None
  """
  # Reading in all rows from the in_file. 
  header, all_comments = read_file_to_list(in_file, True)
  debug(all_comments=all_comments)

  # subreddit_input is a list that contains only the text body. 
  subreddit_input = []
  for count, row in enumerate(all_comments): 
    # print (_text_preprocessing_helper(row[comment_body_col]))
    # print (row[comment_body_col])
    subreddit_input += [_text_preprocessing_helper(row[comment_body_col])]
  debug(subreddit_input=subreddit_input)
  # ['honestly if you are voting for donald trump you are either a white supremacist a bigot or both ive heard people try to say that his hat embodies the ideals of america but no hes just a racist plain and simple im not sure why people are so reluctant to say that if you want an america that is not just for white people then you have to vote for bernie sanders hes the only person left who is standing up for the', 'well im a white supremacist so i guess i have to vote for trump because i want a country only for white people thanks for ex ... and extra items]
  
  # designating all the subreddit classifiers you want to use. You could use 
  # a subset of the 97 subreddit classifiers. ALL_ACTIVE_SUBRREDITS, however, 
  # pulls in all subreddit classifiers. 
  subreddits_model_list = ALL_ACTIVE_SUBREDDITS # DEBUG: TEST_classifier_set
  # Run and outputs all the classifications. 
  final_analysis = run_all_classification(subreddit_input, 
                                          subreddits_model_list)
  debug(final_analysis=final_analysis)
  """
  final_analysis list size: 47 [...]
    item 0:  list size: 97 val: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1]
    item 1:  list size: 97 val: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]
    item 2:  list size: 97 val: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    item 3:  list size: 97 val: [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0]
  """
  # Writes the output of the classifications. Header contains the original 
  # header, followed by the name of the subreddit that the classier belongs to.
  header = header + subreddits_model_list
  output = [header]
  for count, row in enumerate(final_analysis): 
    new_row = all_comments[count] + final_analysis[count]
    output += [new_row]
  write_list_of_list_to_csv(output, out_file)


if __name__ == '__main__':
  # EXAMPLE CLASSIFICATION RUNS: 
  # in_file = "input/" + "poli_100_PR_polite.csv"
  # out_file = "output/" + in_file.split(".")[0].split("/")[-1] + "-out.csv"
  # output_classification(in_file, 2, out_file)

  # in_file = "input/" + "poli_100_baseline.csv"
  # out_file = "output/" + in_file.split(".")[0].split("/")[-1] + "-out.csv"
  # output_classification(in_file, 2, out_file)
  scores = get_classifier_agreement_scores(content)
  print(scores)
  for i in range(content):
    if scores[i] > 80:
      try: 
        print(scores[i], content[i]) 
      except:
        pass
    