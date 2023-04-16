# Generate training data for the model
import os
import math
import pandas as pd
import itertools
import random
import shutil

# load data
filename = 'sentences_augmented10000.xlsx'
try:
    df = pd.read_excel(filename)
except:
    print("File not found!")
    exit()
# get the sentences and coding
sentences = df['Sentences'].tolist()
coding = df['Coding'].tolist()

# find sentences of each leaf node
sentences_dict = {}
for i in range(max(coding)+1):
    sent_list = []
    for j in range(len(sentences)):
        if(coding[j] == i):
            sent_list.append(sentences[j])
    sentences_dict[i] = sent_list

# For each model to be trained, assign the corresponding coding
model_dict = {
    1:[0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1],
    2:[0,0,0,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    3:[-1,-1,-1,-1,-1,0,0,0,1,1,1,2,2,2,2,2],
    4:[0,1,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    5:[-1,-1,-1,0,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    6:[-1,-1,-1,-1,-1,0,1,2,-1,-1,-1,-1,-1,-1,-1,-1],
    7:[-1,-1,-1,-1,-1,-1,-1,-1,0,1,2,-1,-1,-1,-1,-1],
    8:[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,2,3,4],
    }

def generate_data(model_num, size = 2000, percent = 0.8, portion = 0.5):
    # generate training and testing data for the model
    # model_num: the number of the model
    # size: the number of sentences for each leaf node
    # percent: the percentage of training data
    # portion: the portion of the first class (only for model 1 (a bi-partition model))

    coding_list = model_dict[model_num]
    class_num = max(coding_list) + 1
    # sentences related to this model
    sentences_list = []
    # number of sentences for each class (no lesser than size)
    num_list = []
    for i in range(class_num):
        count = 0
        merged_sent_list = []
        indices = [j for j in range(len(coding_list)) if coding_list[j] == i]
        for index in indices:
            merged_sent_list += sentences_dict[index]
            count += len(sentences_dict[index])
        sentences_list.append(merged_sent_list)
        num_list.append(count)
    
    # assertion: the number of total sentences should be no lesser than size
    assert sum(num_list) >= size, "The number of total sentences should be no lesser than size"

    # randomly select sentences for each element of sentences_list
    sents_random_training_list = []
    code_random_training_list = []
    sents_random_testing_list = []
    code_random_testing_list = []
    original_size = size
    for i in range(class_num):
        # generate a random list of [0~size-1]
        if(model_num == 1):
            # proportion (only available for model 1)
            if(i == 0):
                size = math.ceil(original_size * portion)
            else:
                size = math.floor(original_size * (1-portion))
            random_list = list(range(size))
        else:
            # For other models
            random_list = list(range(size//class_num))
        random.shuffle(random_list)
        split_index = int(len(random_list) * 0.8)

        # Split the list into two sublists
        training_list = random_list[:split_index]
        testing_list = random_list[split_index:]
        sents_random_training_list.append([sentences_list[i][j] for j in training_list])
        code_random_training_list.append([i for j in training_list])
        sents_random_testing_list.append([sentences_list[i][j] for j in testing_list])
        code_random_testing_list.append([i for j in testing_list])
    
    #flatten the lists
    sents_random_training_list = list(itertools.chain(*sents_random_training_list))
    code_random_training_list = list(itertools.chain(*code_random_training_list))
    sents_random_testing_list = list(itertools.chain(*sents_random_testing_list))
    code_random_testing_list = list(itertools.chain(*code_random_testing_list))
    data_training = pd.DataFrame({'Sentences':sents_random_training_list, 'Coding':code_random_training_list})
    data_testing = pd.DataFrame({'Sentences':sents_random_testing_list, 'Coding':code_random_testing_list})

    # save the data
    training_path = 'training_model' + str(model_num) + '_size_' + str(original_size) + '_portion_'+str(portion)+'.xlsx'
    testing_path = 'testing_model' + str(model_num) + '_size_' + str(original_size) + '_portion_'+str(portion)+ '.xlsx'
    dir_name = 'model_'+str(model_num)
    # Check if the directory already exists
    if (os.path.exists(dir_name) == False):
        # If the directory doesn't exists, then create it
        os.mkdir(dir_name)
    
    os.chdir(dir_name)

    if os.path.isfile(training_path):
        # If it exists, delete the old file
        os.remove(training_path)
    data_training.to_excel(training_path, index = False)
    if os.path.isfile(testing_path):
        # If it exists, delete the old file
        os.remove(testing_path)
    data_testing.to_excel(testing_path, index = False)

    print("data generated for model " \
          + str(model_num) + " with size: " + str(original_size) + " ;and percentage: " + str(percent)\
            + " ;and proportion: "+str(portion) + "have been saved!\n")
    # Change back to the parent directory
    os.chdir('..')

# model_num = [1,2,3,4,5,6,7,8]
# for i in model_num:
#     generate_data(i,5000,0.8,0.5)


size_list = [500,1000,2000,5000,10000,20000]
portion_list = [0.5,0.6,0.7,0.8]
for size in size_list:
    generate_data(1,size)
    generate_data(2,size)
    generate_data(5,size)
for portion in portion_list:
    generate_data(1,5000,0.8,portion)
    generate_data(2,5000,0.8,portion)
    generate_data(5,5000,0.8,portion)









