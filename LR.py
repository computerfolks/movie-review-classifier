import random
import math

#root directory for the files
root_dir = '/Users/jacobweissman/Desktop/381m project 2/'

def sigmoid(value):
  '''
  calculate the sigmoid
  '''
  neg_value = -1 * value
  return (1 / (1 + math.e ** neg_value))

def cross_entropy_loss(predicted, actual):
  '''
  calculate cross-entrop loss given the predicted value and the actual value
  '''
  # one of these two will be zero, so sum them up at the end to get the return value
  actual_is_one_component = (actual * math.log2(predicted))
  actual_is_zero_component = ((1 - actual) * math.log2(1 - predicted))
  return_value = (-1 * (actual_is_one_component + actual_is_zero_component))
  return return_value

def calc_cross_entropy(weights, feature_vector, actual):
  '''
  calculate the z value, the predicted value based on the z using sigmoid, and the cross entropy loss based on the predicted and actaul
  '''
  z = 0
  for x in range(len(weights)):
    z += weights[x] * feature_vector[x]
  predicted_value = sigmoid(z)
  cross_entropy_loss_value = cross_entropy_loss(predicted_value, actual)
  return cross_entropy_loss_value

def calc_cross_entropy_partial_deriv(weights, feature_vector, actual):
  '''
  calculate the gradient vector using cross-entropy loss
  '''
  z = 0
  for x in range(len(weights)):
    z += weights[x] * feature_vector[x]
  predicted_value = sigmoid(z)
  gradient_vector = []
  for feature_value in feature_vector:
    gradient_vector.append((predicted_value - actual) * feature_value)
  return gradient_vector

def update_weights_gradient_descent(old_weights, feature_vector, actual, learning_rate):
  '''
  update the weight vector using cross-gradient descent results and learning rate
  '''
  gradient_vector = calc_cross_entropy_partial_deriv(old_weights, feature_vector, actual)
  new_weights = []
  for weight_index in range(len(old_weights)):
    new_weights.append(old_weights[weight_index] - learning_rate * gradient_vector[weight_index])
  return new_weights

def LR(train_file, test_file, parameter_file, output_file, learning_rate, user_file_names, wrong_predictions_file):
  '''
  main function
  '''
  #STEP 1: gather training examples and randomize them
  training_examples = []
  with open(train_file) as train:
    for line_with_newline in train:
      line = line_with_newline.strip().replace(' ', '')
      list_line = []
      list_line.append(line[0])
      for x in line[1:]:
        list_line.append(int(x))
      # .append(1) so that each has a 1 value for the b vector
      list_line.append(1)
      training_examples.append(list_line)
  random.shuffle(training_examples)

  #STEP 2: set weights to 0 (weight size is based on size of feature vector + 1 for the b vector to be included)
  # this code is using the b value as just an extra piece of the weight vector, and that position in the examples is always set to 1
  weights = [0] * (len(training_examples[0]) - 1)

  #STEP 3: run the loop

  # current_loss just has to be set above the stopping value to start
  current_loss = 100
  stopping_value = 0.0000001
  training_example_index = 0
  number_of_iterations = 0
  while((current_loss > stopping_value or number_of_iterations < 25000) and number_of_iterations < 50000):
    number_of_iterations += 1
    current_training_example = training_examples[training_example_index]
    if current_training_example[0] == '+':
      actual_value = 1
    else:
      actual_value = 0
    current_loss = calc_cross_entropy(weights, current_training_example[1:], actual_value)
    weights = update_weights_gradient_descent(weights, current_training_example[1:], actual_value, learning_rate)
    training_example_index = (training_example_index + 1) % len(training_examples)
  
  #STEP 4: write parameter results to file
  with open(parameter_file, 'w') as parameter:
    for weight in weights:
      print(f'{weight:.3f}', file=parameter)

  #STEP 5: get testing data
  with open(test_file) as test:
    output = open(output_file, 'w')
    testing_examples = []
    for line_with_newline in test:
      line = line_with_newline.strip().replace(' ', '')
      list_test_line = []
      list_test_line.append(line[0])
      for x in line[1:]:
        list_test_line.append(int(x))
      # .append(1) so that each has a 1 value for the b vector
      list_test_line.append(1)
      testing_examples.append(list_test_line)
  
  #STEP 6: test on testing data
  # to test, calculate z. if positive, guess positive.
  correct_guesses = 0
  total_guesses = 0

  #keep track of which file corresponds to which guess
  filenames = user_file_names
  files = []
  with open(filenames) as file_names:
    for line_with_newline in file_names:
      line = line_with_newline.strip().replace(' ', '')
      files.append(line)

  #file for just wrong guesses to investigate
  wrong_predictions = open(wrong_predictions_file, 'w')
  wrong_prediction_tuples = []

  for test_example in testing_examples:
    feature_vector = test_example[1:]
    actual = test_example[0]
    z = 0
    for x in range(len(weights)):
      z += weights[x] * feature_vector[x]
    # equivalent to checking if sigmoid is > 0.5
    if z > 0:
      guess = '+'
    else:
      guess = '-'
    
    if guess == actual:
      correct_guesses += 1
    total_guesses += 1

    print(files[total_guesses - 1], file = output, end = ' ')
    # print(test_example[1:], file = output, end = ' ')
    print("GUESS: " + guess + " ACTUAL: " + actual, file = output, end = ' ')
    print(file = output)
    if guess != actual:
      # we guessed wrong, so our confidence in the actual is 1 - sigmoid
      if z <= 0:
        confidence_in_correct_guess = sigmoid(z)
      else:
        confidence_in_correct_guess = 1 - sigmoid(z)
      wrong_prediction_tuples.append((files[total_guesses - 1], test_example[1:], "GUESS: " + guess + " ACTUAL: " + actual, confidence_in_correct_guess))
  

  print("ACCURACY: ", end = '', file = output)
  accuracy = correct_guesses / total_guesses
  print(f'{accuracy : .3f}', file = output)
  print(f'{accuracy : .3f}')
  output.close()

  #sort through wrong predictions by lowest probability assigned to actual answer, to make it easier to investigate
  wrong_prediction_tuples = sorted(wrong_prediction_tuples, key = lambda x : x[3])
  for wrong_prediction_tuple in wrong_prediction_tuples:
    print(wrong_prediction_tuple[0], file = wrong_predictions)
    print(wrong_prediction_tuple[1], file = wrong_predictions)
    print(wrong_prediction_tuple[2], file = wrong_predictions)
    print(f"PROBABILITY ASSIGNED TO ACTUAL ANSWER: {wrong_prediction_tuple[3]:.7f}", file = wrong_predictions)
    print(file = wrong_predictions)
  
  wrong_predictions.close()


train_file = root_dir + 'train_vec_output.txt'
test_file = root_dir + 'test_vec_output.txt'
# parameter_file = root_dir + 'parameters.txt'
parameter_file = root_dir + 'movie-review-BOW.LR'
output_file = root_dir + 'predictions.txt'
filenames = root_dir + "file_names.txt"
wrong_predictions_file = root_dir + "wrong_predictions.txt"
learning_rate = 0.15
LR(train_file, test_file, parameter_file, output_file, learning_rate, filenames, wrong_predictions_file)

