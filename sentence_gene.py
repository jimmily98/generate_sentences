import pandas as pd
import itertools

excel_file = 'sentences.xlsx'
sheets_dict = pd.read_excel(excel_file, sheet_name=None)
sheet_names = list(sheets_dict.keys())

# process sheet1
sheet1 = sheets_dict[sheet_names[0]]
sents_list = []
for i in range(sheet1.shape[0]):
    sents_node = sheet1['templates'].values[i].split('\n')
    sents_list.append(sents_node)

# process sheet2
sheet2 = sheets_dict[sheet_names[1]]
pattern_list = sheet2.keys()
pattern_dict = {}
for i in range(sheet2.shape[1]):
    temp = list(sheet2[pattern_list[i]].values)
    # drop nan
    temp = [x for x in temp if str(x) != 'nan']
    for j in range(len(temp)):
        temp[j] = str(temp[j])
    pattern_dict[pattern_list[i]] = temp

# generate sentences
# replace the {} with the corresponding pattern lists
# sentence counter
count = 0
coding_list = []
node_num = 0
replace_sents_list = []

for sents_node in sents_list:
    new_list = []
    for sent in sents_node:
        # delete the brackets 
        sent = sent.replace('{', '').replace('}', '')
        # temp is the list of sentences after one replacement
        # initialize temp with the original sentence
        temp = [sent]
        for key, value in pattern_dict.items():
            if (key in sent):
                new_temp = []
                for temp_sent in temp:
                    # reconstruct the sentence list (temp) after one replacement
                    for pattern in pattern_dict[key]:
                        new_temp.append(temp_sent.replace(key, pattern))
                temp = new_temp
        count += len(temp)
        new_list.append(temp)
        coding_list.append([node_num]*len(temp))
    replace_sents_list.append(new_list)
    node_num += 1

# flatten the list
replace_sents_list = list(itertools.chain.from_iterable(itertools.chain.from_iterable(replace_sents_list)))
coding_list = list(itertools.chain.from_iterable(coding_list))
data = {'Sentences':replace_sents_list, 'Coding':coding_list}
df = pd.DataFrame(data)
df.to_excel('sentences_generated.xlsx', index = False)

print('end of sentence generation!')