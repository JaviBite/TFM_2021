
import json
import sys
import codecs
import re
import random, string

FRAMES_PER_SECOND = 25

def lookforaction(data, id_recipe, id_step):

    steps = data['data']['inserts']['steps']
    action = "Not found"
    for value in steps:
        if id_recipe == value['id_recipe'] and id_step == value['orderstep']:
            action = value['name']
            break

    return action
  
file_ours = sys.argv[1]
file_bsh = sys.argv[2]

outfile = "out.json"

# Opening JSON file
fours = open(file_ours, encoding='utf-8')
fbsh = open(file_bsh, encoding='utf-8')
  
# returns JSON object as 
# a dictionary
data_ours = json.load(fours)
data_bsh = json.load(fbsh)
  
# Iterating through the json
print("Getting data...")

video_nameid = re.search("[0-9A-Z]{12}_[0-9]{4}_[0-9]{2}_[0-9]{2}", file_bsh)[0]

actions = []
json_action_list = data_bsh['data']['inserts']['recipedata']
for value in json_action_list:

    id_recipe = value['id_recipe']
    id_step = value['id_step']
    init_time = int(value['timestart']) / 10000
    if value['timeend'] is not None:
        end_time = int(value['timeend']) / 10000
    else:
        end_time = -1

    str_action = lookforaction(data_bsh, id_recipe,id_step)

    actions.append({'time':[init_time,end_time], 'act':str_action})

print("Look for video on dataset...")

vid = -1
for value in data_ours['file'].values():
    video_name = re.search("[0-9A-Z]{12}_[0-9]{4}_[0-9]{2}_[0-9]{2}", value['fname'])[0]
    if video_name == video_nameid:
        vid = value['fid']
        print("Video found, ID: ", vid)
        break

if vid == -1:
    print("Video not found, closing...")
    exit()

print("Adding new metadata...")

for action in actions:

    time_start = action['time'][0]
    time_end = action['time'][1]
    if time_end < 0:
        time_end = time_start + 5

    # Generating the action value
    new_value = {'vid':str(vid), 
                 'flg':0, 
                 'z':[time_start, time_end], 
                 'xy':[], 
                 'av':{'1':action['act']}}

    # Generating random metadata id
    randid = ''.join(random.choices(string.ascii_lowercase + string.ascii_uppercase + string.digits, k=8))
    new_key = '30_' + randid

    # Inserting the new data
    data_ours['metadata'][new_key] = new_value



for i in ["1","2","3","4"]:
    data_ours['config']['file']['loc_prefix'][i] = "file:///E:/tfm/actions/videos"

print("Dumping data...")
with codecs.open(outfile, 'w', encoding='utf-8') as out:
    json.dump(data_ours, out, indent=None, ensure_ascii=False)
  
print("All ok!")

# Closing file
fours.close()
fbsh.close()
out.close()