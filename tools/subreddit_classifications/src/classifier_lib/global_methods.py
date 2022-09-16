import random
import string
import csv
import time
import datetime as dt
import pathlib
import os
import sys
import numpy
import math

from os import listdir
from collections import Counter 

from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import t
from scipy import stats


def create_folder_if_not_there(curr_path): 
  """
  Checks if a folder in the curr_path exists. If it does not exist, creates
  the folder. 
  Note that if the curr_path designates a file location, it will operate on 
  the folder that contains the file. But the function also works even if the 
  path designates to just a folder. 
  Args:
    curr_list: list to write. The list comes in the following form:
               [['key1', 'val1-1', 'val1-2'...],
                ['key2', 'val2-1', 'val2-2'...],]
    outfile: name of the csv file to write    
  RETURNS: 
    True: if a new folder is created
    False: if a new folder is not created
  """
  outfolder_name = curr_path.split("/")
  if len(outfolder_name) != 1: 
    # This checks if the curr path is a file or a folder. 
    if "." in outfolder_name[-1]: 
      outfolder_name = outfolder_name[:-1]

    outfolder_name = "/".join(outfolder_name)
    if not os.path.exists(outfolder_name):
      os.makedirs(outfolder_name)
      return True

  return False 


def write_list_of_list_to_csv(curr_list_of_list, outfile):
  """
  Writes a list of list to csv. 
  Unlike write_list_to_csv_line, it writes the entire csv in one shot. 
  ARGS:
    curr_list_of_list: list to write. The list comes in the following form:
               [['key1', 'val1-1', 'val1-2'...],
                ['key2', 'val2-1', 'val2-2'...],]
    outfile: name of the csv file to write    
  RETURNS: 
    None
  """
  create_folder_if_not_there(outfile)
  with open(outfile, "w") as f:
    writer = csv.writer(f)
    writer.writerows(curr_list_of_list)


def write_list_to_csv_line(line_list, outfile): 
  """
  Writes one line to a csv file.
  Unlike write_list_of_list_to_csv, this opens an existing outfile and then 
  appends a line to that file. 
  This also works if the file does not exist already. 
  ARGS:
    curr_list: list to write. The list comes in the following form:
               ['key1', 'val1-1', 'val1-2'...]
               Importantly, this is NOT a list of list. 
    outfile: name of the csv file to write   
  RETURNS: 
    None
  """
  create_folder_if_not_there(outfile)

  # Opening the file first so we can write incrementally as we progress
  curr_file = open(outfile, 'a',)
  csvfile_1 = csv.writer(curr_file)
  csvfile_1.writerow(line_list)
  curr_file.close()


def read_file_to_list(curr_file, header=False): 
  """
  Reads in a csv file to a list of list. If header is True, it returns a 
  tuple with (header row, all rows)
  ARGS:
    curr_file: path to the current csv file. 
  RETURNS: 
    List of list where the component lists are the rows of the file. 
  """
  if not header: 
    analysis_list = []
    with open(curr_file) as f_analysis_file: 
      data_reader = csv.reader(f_analysis_file, delimiter=",")
      for count, row in enumerate(data_reader): 
        analysis_list += [row]
    return analysis_list
  else: 
    analysis_list = []
    with open(curr_file) as f_analysis_file: 
      data_reader = csv.reader(f_analysis_file, delimiter=",")
      for count, row in enumerate(data_reader): 
        analysis_list += [row]
    return analysis_list[0], analysis_list[1:]


def read_file_to_set(curr_file, col=0): 
  """
  Reads in a "single column" of a csv file to a set. 
  ARGS:
    curr_file: path to the current csv file. 
  RETURNS: 
    Set with all items in a single column of a csv file. 
  """
  analysis_set = set()
  with open(curr_file) as f_analysis_file: 
    data_reader = csv.reader(f_analysis_file, delimiter=",")
    for count, row in enumerate(data_reader): 
      analysis_set.add(row[col])
  return analysis_set


def get_row_len(curr_file): 
  """
  Get the number of rows in a csv file 
  ARGS:
    curr_file: path to the current csv file. 
  RETURNS: 
    The number of rows
    False if the file does not exist
  """
  try: 
    analysis_set = set()
    with open(curr_file) as f_analysis_file: 
      data_reader = csv.reader(f_analysis_file, delimiter=",")
      for count, row in enumerate(data_reader): 
        analysis_set.add(row[0])
    return len(analysis_set)
  except: 
    return False


def check_if_file_exists(curr_file): 
  """
  Checks if a file exists
  ARGS:
    curr_file: path to the current csv file. 
  RETURNS: 
    True if the file exists
    False if the file does not exist
  """
  try: 
    with open(curr_file) as f_analysis_file: pass
    return True
  except: 
    return False


def find_filenames(path_to_dir, suffix=".csv"):
  """
  Given a directory, find all files that ends with the provided suffix and 
  returns their paths.  
  ARGS:
    path_to_dir: Path to the current directory 
    suffix: The target suffix.
  RETURNS: 
    A list of paths to all files in the directory. 
  """
  filenames = listdir(path_to_dir)
  return [ path_to_dir+"/"+filename 
           for filename in filenames if filename.endswith( suffix ) ]


def average(list_of_val): 
  """
  Finds the average of the numbers in a list.
  ARGS:
    list_of_val: a list of numeric values  
  RETURNS: 
    The average of the values
  """
  return sum(list_of_val)/float(len(list_of_val))


def std(list_of_val): 
  """
  Finds the std of the numbers in a list.
  ARGS:
    list_of_val: a list of numeric values  
  RETURNS: 
    The std of the values
  """
  std = numpy.std(list_of_val)
  return std


def get_random_alphanumeric_string(length):
  """
  Returns a randomly generated alphanumeric string with a given length. 
  ARGS:
    length: the length of the alphanumeric string.  
  RETURNS: 
    A randomly generated string. 
  """
  letters_and_digits = string.ascii_letters + string.digits
  result_str = ''.join((random.choice(letters_and_digits) for i in range(length)))
  return result_str


def most_frequent(List): 
  """
  Given a list, find th element that is most frequently occuring. 
  ARGS:
    List: The input list.  
  RETURNS: 
    Element that is most frequently occuring.
  """
  occurence_count = Counter(List) 
  return occurence_count.most_common(1)[0][0] 


def unix_time_to_readable_time(unix_time): 
  """
  Converts unix time to a readable time str. Unix time is just an int that 
  represents some time. Often used by reddit (praw).  
  ARGS:
    List: unix time.  
  RETURNS: 
    A readable time str.
  """
  return dt.datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')


def cronbach_alpha(itemscores):
  """
  Given a list that contains two lists of the same length, calculates the 
  cronbach alpha measure that captures the internal reliability. 
  ARGS:
    List: A list of two lists of the same length (eg. [[1,0,1,0,0,0], 
                                                       [1,0,0,0,0,0]])
  RETURNS: 
    cronbach_alpha in float.
  """
  itemscores = numpy.asarray(itemscores)
  itemvars = itemscores.var(axis=1, ddof=1)
  tscores = itemscores.sum(axis=0)
  nitems = len(itemscores)

  return nitems / (nitems-1.) * (1 - itemvars.sum() / tscores.var(ddof=1))

 
def f1_related_stats(itemscore): 
  """
  Given a list that contains two lists of the same length, calculates the 
  f1 score as well as the precision and recall. 
  ARGS:
    List: A list of two lists of the same length (eg. [[1,0,1,0,0,0], 
                                                       [1,0,0,0,0,0]])
  RETURNS: 
    f1, precision and recall
  """
  list_1 = itemscore[0]
  list_2 = itemscore[1]

  tp, fp, fn = 0, 0, 0

  for count, i in enumerate(list_1): 
    if str(list_1[count]) == str(1) and str(list_2[count]) == str(1): 
      tp += 1
    elif str(list_1[count]) == str(0) and str(list_2[count]) == str(1):  
      fp += 1
    elif str(list_1[count]) == str(1) and str(list_2[count]) == str(0):  
      fn += 1

  if tp+fp != 0: 
    precision = float(tp)/(tp+fp)
  else: 
    precision = "UD"

  if tp+fn != 0: 
    recall = float(tp)/(tp+fn)
  else: 
    recall = "UD"

  if precision != "UD" and recall  != "UD": 
    if precision+recall != 0: 
      f1 = 2 * ((precision*recall)/(precision+recall))
    else: 
      f1 = "UD"
  else: 
    f1 = "UD"

  return f1, precision, recall



def pad_list(curr_list, wanted_len, pad_char="-"): 
  """
  Given a list and a wanted length, pad the list. 
  ARGS:
    List: A list you want to pad
    wanted len: Length you want it to be
    pad_char: the character you want to use to pad the list
  RETURNS: 
    f1, precision and recall
  """
  while len(curr_list) < wanted_len: 
    curr_list += [pad_char]
  return curr_list



# function for calculating the t-test for two dependent samples
# Also known as student's ttest... assumes equal sample size
# https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/
def dependent_ttest(data1, data2, alpha):
  data1 = numpy.array(data1)
  data2 = numpy.array(data2)

  # calculate means
  mean1, mean2 = mean(data1), mean(data2)
  # number of paired samples
  n = len(data1)
  # sum squared difference between observations
  d1 = sum([(data1[i]-data2[i])**2 for i in range(n)])
  # sum difference between observations
  d2 = sum([data1[i]-data2[i] for i in range(n)])
  # standard deviation of the difference between means
  sd = sqrt((d1 - (d2**2 / n)) / (n - 1))
  # standard error of the difference between the means
  sed = sd / sqrt(n)
  # calculate the t statistic
  t_stat = (mean1 - mean2) / sed
  # degrees of freedom
  df = n - 1
  # calculate the critical value
  cv = t.ppf(1.0 - alpha, df)
  # calculate the p-value
  p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
  # return everything
  return t_stat, df, cv, p


def welch_ttest(data1, data2): 
  df = len(data1) + len(data2) - 2
  return stats.ttest_ind(data1, data2, equal_var = False), df


def trim_outlier(curr_list): 
  curr_average = average(curr_list)
  curr_std = std(curr_list)
  min_threshold = curr_average - (curr_std * 2)
  max_threshold = curr_average + (curr_std * 2)

  new_list = []
  for i in curr_list: 
    if i < max_threshold and i > min_threshold: 
      new_list += [i]

  return new_list 


def get_random_binary_pick(positive_probability): 
  curr_random_axis = random.uniform(0.0, 1.0)
  if positive_probability >= curr_random_axis: 
    return 1
  else: 
    return 0  


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * numpy.array(data)
    n = len(a)
    m, se = numpy.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h  



def percentile(N, percent, key=lambda x:x):
    """
    Find the percentile of a list of values.
    @parameter N - is a list of values. Note N MUST BE already sorted.
    @parameter percent - a float value from 0.0 to 1.0.
    @parameter key - optional key function to compute value from each element of N.
    @return - the percentile of the values
    """
    N.sort()
    if not N:
        return None
    k = (len(N)-1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(N[int(k)])
    d0 = key(N[int(f)]) * (c-k)
    d1 = key(N[int(c)]) * (k-f)
    return d0+d1


def bootstrap_confidence_interval(data, confidence=0.95):
  bottom_percentile_mark = (1 - confidence)/float(2)
  top_percentile_mark = 1 - bottom_percentile_mark

  p_1 = percentile(data, bottom_percentile_mark)
  p_2 = percentile(data, top_percentile_mark)

  return p_1, p_2


if __name__ == '__main__':
  data1 = [90, 85, 88, 89, 94, 91, 79, 83, 87, 88, 91, 90]
  data2 = [67, 90, 71, 95, 88, 83, 72, 66, 75, 86, 93, 84, 10]
  print(mean_confidence_interval(data1, confidence=0.95))
  # alpha = 0.05
  # # print (dependent_ttest(data1, data2, alpha))
  # print (welch_ttest(data1, data2))




