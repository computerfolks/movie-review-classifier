import os
import string

#root directory for the files
root_dir = '/Users/jacobweissman/Desktop/381m project 2/'
training_file_dir = '/Users/jacobweissman/Desktop/381m project 2/'

#STEP 1: DETERMINE WHICH WORDS TO USE BY FINDING RELATIVE FREQUENCIES BETWEEN POS AND NEG IN TRAINING DATA

def generate_frequencies(neg_output_file, neg_directory_file, pos_output_file, pos_directory_file, diff_output_file, total_output_file):
  '''
  generate word frequencies for files, ignoring stop words
  generate negative output file (negative word distribution), positive output file, total output file (total word distribution), and relative output file (file of relative frequencies)
  '''
  #get stop words
  stop_words = set()
  # https://gist.github.com/sebleier/554280
  stop_words_directory = root_dir + 'stopwords.txt'
  with open(stop_words_directory) as stop_words_file:
    for line in stop_words_file:
      word = line.strip()
      stop_words.add(word)

  def file_to_word_list(directory):
    '''
    convert a set of files to a list of word:frequency pairs
    note: each file can only contribute to frequency once for each word
    '''
    number_of_files = 0

    #iterate over all files in the directory
    words_seen = set()
    word_count = {}
    table = str.maketrans('', '', string.punctuation)
    for filename in os.listdir(directory):
      f = os.path.join(directory, filename)
      number_of_files += 1
      # to avoid Mac OS created files
      if filename == '.DS_Store':
        continue
      # if number_of_neg_files > 1000:
      #   break
      with open(f) as file:
        #get input into a string
        file_string = file.read()
        words = file_string.lower().replace('<br />', '').translate(table).split()
        for word in words:
          #ignore stop words
          if word in stop_words:
            continue
          #ignore if word has been seen this iteration (this file)
          if word in words_seen:
            continue

          # add to seen words
          words_seen.add(word)

          #increment count
          current_count = word_count.get(word, 0)
          current_count += 1
          word_count[word] = current_count

        #clear set for next round
        words_seen.clear()

    #sort words based on highest frequency
    words_seen_list = [(word, amount) for word, amount in word_count.items()]
    words_seen_list_sorted = sorted(words_seen_list, key = lambda x : x[1], reverse = True)
    return number_of_files, words_seen_list_sorted, word_count
              
  #output neg results to neg output file
  neg_output = open(neg_output_file, 'w')
  neg_directory = neg_directory_file
  neg_number_of_files, neg_word_frequency, neg_word_count = file_to_word_list(neg_directory)
  for word, amount in neg_word_frequency:
    print(word + ": " + str(amount), file=neg_output)

  #output pos results to pos output file
  pos_output = open(pos_output_file, 'w')
  pos_directory = pos_directory_file
  pos_number_of_files, pos_word_frequency, pos_word_count = file_to_word_list(pos_directory)
  for word, amount in pos_word_frequency:
    print(word + ": " + str(amount), file=pos_output)

  neg_output.close()
  pos_output.close()

  difference_amount = {}
  total_amount = {}
  for word, neg_word_amount in neg_word_frequency:
    pos_word_amount = pos_word_count.get(word, 0)
    difference_amount[word] = pos_word_amount - neg_word_amount
    total_amount[word] = pos_word_amount + neg_word_amount

  # OPTIONAL: slows things down a lot, will not have significant impact because these words will barely appear
  # get totals for words that only appear in positive reviews
  # for word, pos_word_amount in pos_word_frequency:
  #   if word not in total_amount.keys():
  #     difference_amount[word] = pos_word_amount
  #     total_amount[word] = pos_word_amount

  difference_amount_list = [(word, amount) for word, amount in difference_amount.items()]
  difference_amount_list_sorted = sorted(difference_amount_list, key = lambda x : abs(x[1]), reverse = True)
  diff_output = open(diff_output_file, 'w')
  for word, amount in difference_amount_list_sorted:
    print(word + ": " + str(amount), file=diff_output)

  total_amount_list = [(word, amount) for word, amount in total_amount.items()]
  total_amount_list_sorted = sorted(total_amount_list, key = lambda x : abs(x[1]), reverse = True)
  total_output = open(total_output_file, 'w')
  for word, amount in total_amount_list_sorted:
    print(word + ": " + str(amount), file=total_output)
    
  diff_output.close()
  total_output.close()

# STEP 2: generate the bag of words to be used
def get_bag_of_words(number_of_words, frequencies_file):
  '''
  get most frequent words (or word differentials) from a sorted file with word frequencies
  '''
  n = 0
  bag_of_words_list = []
  frequencies = open(frequencies_file)
  for line in frequencies:
    if n == number_of_words:
      break
    n += 1
    word = line.split(':')[0]
    bag_of_words_list.append(word)
  frequencies.close()
  return bag_of_words_list

def clear_files(*args):
  '''
  clear a list of files
  '''
  for file in args:
    clear_file = open(file, 'w')
    clear_file.close()  


def generate_feature_vectors(positive_or_negative, train_or_test, bag_of_words_list, vec_output_file, directory, net_output_file, file_names):
  '''
  generate feature vector files and file which stores file names
  '''
  vec_output = open(vec_output_file, 'w')
  net_output = open(net_output_file, 'a')

  #iterate over all files in the directory
  bag_of_words_seen = {}
  for word in bag_of_words_list:
    bag_of_words_seen[word] = 0
  table = str.maketrans('', '', string.punctuation)
  if train_or_test == "test":
    file_name_output = open(file_names, 'a')
  for filename in os.listdir(directory):
    if train_or_test == "test":
      print(filename, file = file_name_output)
    f = os.path.join(directory, filename)
    for word in bag_of_words_seen:
      bag_of_words_seen[word] = 0
    with open(f) as file:
      if filename == '.DS_Store':
        continue
      #get input into a string
      file_string = file.read()
      words = file_string.lower().replace('<br />', '').translate(table).split()
      for word in words:
        if word in bag_of_words_list:
          bag_of_words_seen[word] = 1
    
    #print the result
    if positive_or_negative == "positive":
      print('+ ', file = vec_output, end = '')
      print('+ ', file = net_output, end = '')
    else:
      print('- ', file = vec_output, end = '')
      print('- ', file = net_output, end = '')

    #print the features
    vector_list = [(word, bool_value) for word, bool_value in bag_of_words_seen.items()]
    vector_list_sorted = sorted(vector_list, key = lambda x : x[0])
    for word, bool_value in vector_list_sorted:
      print(str(bool_value) + ' ', file = vec_output, end = '')
      print(str(bool_value) + ' ', file = net_output, end = '')
    print(file = vec_output)
    print(file = net_output)

  vec_output.close()
  if train_or_test == "test":
    file_name_output.close()

# # STEP 1
neg_output = root_dir + 'neg_word_frequency.txt'
neg_directory = training_file_dir + 'movie-review-HW2/aclImdb/train/neg'
pos_output = root_dir + 'pos_word_frequency.txt'
pos_directory = training_file_dir + 'movie-review-HW2/aclImdb/train/pos'
diff_output = root_dir + 'diff_word_frequency.txt'
total_output = root_dir + 'total_word_frequency.txt'
generate_frequencies(neg_output, neg_directory, pos_output, pos_directory, diff_output, total_output)

# # STEP 2: get the bag of words to be used
# # pick the number of words to use
number_of_words = 200
bag_of_words_list = sorted(get_bag_of_words(number_of_words, diff_output))
print(bag_of_words_list)

#STEP pre-3: clear the files
file_1 = root_dir + 'train_vec_output.txt'
file_2 = root_dir + 'test_vec_output.txt'
file_3 = root_dir + 'file_names.txt'
clear_files(file_1, file_2, file_3)

#STEP 3: generate vector representations for each word for the training data
pos_train_output = root_dir + 'pos_train_vec_output.txt'
neg_train_output = root_dir + 'neg_train_vec_output.txt'
pos_train_directory = training_file_dir + 'movie-review-HW2/aclImdb/train/pos'
neg_train_directory = training_file_dir + 'movie-review-HW2/aclImdb/train/neg'
net_train_output = root_dir + 'train_vec_output.txt'
file_names = root_dir + 'file_names.txt'
generate_feature_vectors('positive', 'train', bag_of_words_list, pos_train_output, pos_train_directory, net_train_output, file_names)
generate_feature_vectors('negative', 'train', bag_of_words_list, neg_train_output, neg_train_directory, net_train_output, file_names)

#STEP 4: generate vector representations for each word for the test data
pos_test_output = root_dir + 'pos_test_vec_output.txt'
neg_test_output = root_dir + 'neg_test_vec_output.txt'
pos_test_directory = training_file_dir + 'movie-review-HW2/aclImdb/test/pos'
neg_test_directory = training_file_dir + 'movie-review-HW2/aclImdb/test/neg'
net_test_output = root_dir + 'test_vec_output.txt'

generate_feature_vectors('positive', 'test', bag_of_words_list, pos_test_output, pos_test_directory, net_test_output, file_names)
generate_feature_vectors('negative', 'test', bag_of_words_list, neg_test_output, neg_test_directory, net_test_output, file_names)