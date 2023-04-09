#imported libraries
import pandas as pd
import random as rd
import string
import openpyxl 

#function reading the excel file 
def read_file(file_name):
    df = pd.read_excel(file_name)
    return df

#function converting to a spell-error sentence
def convert_sentence(sentence):
    lower_bound = min(5, len(sentence)//5)
    number_of_errors = rd.randint(1, lower_bound)
    
    for i in range(number_of_errors):
        type_of_error = rd.randint(1, 3)
        #type 1: adding a new character
        if (type_of_error == 1):
            add_character = rd.choice(string.ascii_letters)
            position_of_error = rd.randint(0, len(sentence))
            if (position_of_error == 0): 
                sentence = add_character + sentence
            else: 
                sentence = sentence[:position_of_error-1] + add_character + sentence[position_of_error-1:]
        #type 2: remove a character
        if (type_of_error == 2):
            position_of_error = rd.randint(0, len(sentence)-1)
            sentence = sentence[:position_of_error] + sentence[position_of_error+1:]
        #type 3: change a character to another character
        if (type_of_error == 3):
            replace_character = rd.choice(string.ascii_letters)
            position_of_error = rd.randint(0, len(sentence)-1)
            sentence = sentence[:position_of_error] + replace_character + sentence[position_of_error+1:]
    #return an error sentence
    return sentence
    
#function writing to a excel file 
def writing_file(data, file_name):
    error_file_name = file_name + "_error.xlsx"
    data.to_excel(error_file_name, sheet_name='Sheet 1', index=False, header=False)


#main function
#sentence = "I love you!"
#print(convert_sentence(sentence))
file_name = 'sentences_generated.xlsx'
data = read_file(file_name)
df = data['Sentences']
#print(convert_sentence(df[1]))
#df = convert_sentence(df)
for i in range(len(df)):
    if (isinstance(df[i], str)):
        df[i] = convert_sentence(df[i])
writing_file(df,file_name)