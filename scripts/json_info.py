
import json
import sys
import codecs, unicodedata

def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)
  
file1 = sys.argv[1]

# Opening JSON file
f1 = open(file1, encoding='utf-8')
  
# returns JSON object as 
# a dictionary
data1 = json.load(f1)

actions = {}
for tag, value in zip(data1['metadata'], data1['metadata'].values()):
    #print(tag, value)

    #action = strip_accents(str(value['av']['1'])).lower()
    action = str(value['av']['1'])

    #if action[-1] == 's':
    #    action = action[0:-1]

    if action in actions.keys():
        actions[action] += 1
    else:
        actions[action]  = 1

actions = dict(sorted(actions.items(), reverse=True, key=lambda item: item[1]))
#actions = dict(sorted(actions.items()))

just_actions = {}
just_nouns = {}
for action, times in zip(actions, actions.values()):
        split_actions = action.split()
        just_action = split_actions[0] # 0 action 1 ingrediente
        just_noun = split_actions[1]

        if just_action in just_actions.keys():
            just_actions[just_action] += times
        else:
            just_actions[just_action]  = times

        if just_noun in just_nouns.keys():
            just_nouns[just_noun] += times
        else:
            just_nouns[just_noun]  = times

just_actions = dict(sorted(just_actions.items(), reverse=True, key=lambda item: item[1]))
just_nouns = dict(sorted(just_nouns.items(), reverse=True, key=lambda item: item[1]))

#for action, times in zip(just_actions, just_actions.values()):          
#    print(action, " -> ", times)  

frec1 = 0
with codecs.open("info.txt", 'w', encoding='utf-8') as out:
    for action, times in zip(actions, actions.values()):
        if times <= 1:
            frec1 += 1

          
        #print(action, " -> ", times)  
        out.write(action + " -> " + str(times) + "\n")

#print("Acciones con frecuencia 1: ", frec1)

# Closing file
f1.close()