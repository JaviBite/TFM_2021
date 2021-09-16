import json
import sys
import codecs

filepath = sys.argv[1]
f = open(filepath, encoding='utf-8')
data = json.load(f)

for i in ["1","2","3","4"]:
    data['config']['file']['loc_prefix'][i] = "file:///%DIR_TO_VIDEOS%"

outfile = 'out.json'
numlines = 0

for line in open(filepath): 
    numlines += 1
    if numlines > 5:
        break

f.close()

print("Num lines: ", numlines)
json_indent = None

if numlines <= 1:
    json_indent = 4
    print("Expaning json with indent = ", json_indent)
else:
    print("Compressing json with no indent, 1 line")





with codecs.open(outfile, 'w', encoding='utf-8') as out:
    json.dump(data, out, indent=json_indent, ensure_ascii=False)


