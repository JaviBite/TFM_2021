
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
file2 = sys.argv[2]

outfile = "out.json"

# Opening JSON file
f1 = open(file1, encoding='utf-8')
f2 = open(file2, encoding='utf-8')
  
# returns JSON object as 
# a dictionary
data1 = json.load(f1)
data2 = json.load(f2)
  
# Iterating through the json
print("Fusion data...")

actions = []
for tag, value in zip(data1['metadata'], data1['metadata'].values()):
    actions.append([tag, value])

for tag, value in actions:
    data2['metadata'][tag] = value


print("Parsing and cleaning...")

for tag, value in zip(data2['metadata'], data2['metadata'].values()):

    # No accents and lowercase
    action = strip_accents(str(value['av']['1'])).lower()

    # No plurals
    if action[-1] == 's':
        action = action[0:-1]

    value['av']['1'] = action
    data2['metadata'][tag] = value



for i in ["1","2","3","4"]:
    data2['config']['file']['loc_prefix'][i] = "file:///%DIR_TO_VIDEOS%"

print("Dumping data...")
with codecs.open(outfile, 'w', encoding='utf-8') as out:
    json.dump(data2, out, indent=None, ensure_ascii=False)
  
print("All ok!")

# Closing file
f1.close()
f2.close()
out.close()