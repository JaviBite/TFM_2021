
import json
import sys
import codecs
import re
import random, string
import glob, os

FRAMES_PER_SECOND = 25
REGEXP_VIDEO_NAME = "[0-9A-Z]{12}_[0-9]{4}_[0-9]{2}_[0-9]{2}"

import argparse

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("via_json", type=str, help="Path to the json of the dataset (via annotations)")
    parser.add_argument('bsh_dir_json', help="Dir where the video bsh jsons are")
    parser.add_argument('out_json', nargs='?', default="out.json", help="Path file for the output json")
    parser.add_argument('-v','--verbose', help="Show prints on screen", action="store_true")
    # parser.add_argument("-m","--mode", default='full', help="Info to excract", choices=['full', 'actions', 'noums'])
    # parser.add_argument("-sm","--sort_mode", default='frec', help="Sorting mode, alphabetical or frequency", choices=['alph', 'frec'])
    # parser.add_argument("-so","--sort_order", default='desc', help="Sorting order (by frequency or alph)", choices=['asc', 'desc'])

    return parser.parse_args()

def lookforaction(data, id_recipe, id_step):

    steps = data['data']['inserts']['steps']
    action = "Not found"
    for value in steps:
        if id_recipe == value['id_recipe'] and id_step == value['orderstep']:
            action = value['name']
            break

    return action
  

def main():

    args = parseargs()

    file_ours = args.via_json
    dir_bsh = args.bsh_dir_json

    outfile = args.out_json

    verbose = args.verbose

    # Opening JSON file
    fours = open(file_ours, encoding='utf-8')
    data_ours = json.load(fours)
    fours.close()

    modified_videos = []

    # Open each json of target directory
    for file_bsh in glob.glob(dir_bsh + "/*.json"):

        fbsh = open(file_bsh, encoding='utf-8')
        data_bsh = json.load(fbsh)
        fbsh.close()
        
        # Iterating through the json
        if verbose: print("Getting data...")

        video_nameid = re.search(REGEXP_VIDEO_NAME, file_bsh)

        if video_nameid is None:
            eprint("File \"", file_bsh, "\" is bad named, skipping...")
            break

        video_nameid = video_nameid[0]

        if verbose: print("Processing video: ", video_nameid)

        actions = []
        json_action_list = data_bsh['data']['inserts']['recipedata']
        for value in json_action_list:

            id_recipe = value['id_recipe']
            id_step = value['id_step']
            init_time = int(value['timestart']) / 10000 #TODO
            if value['timeend'] is not None:
                end_time = int(value['timeend']) / 10000  #TODO
            else:
                end_time = -1

            str_action = lookforaction(data_bsh, id_recipe,id_step)

            actions.append({'time':[init_time,end_time], 'act':str_action})

        if verbose: print("\tLook for video on dataset...")

        vid = -1
        for value in data_ours['file'].values():
            video_name = re.search(REGEXP_VIDEO_NAME, value['fname'])[0]
            if video_name == video_nameid:
                vid = value['fid']
                if verbose: print("\tVideo found, ID: ", vid)
                break

        if vid == -1:
            eprint("Video \"", video_nameid, "\" not found, skipping...")
            break

        if verbose: print("\tAdding new metadata...")

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
            new_key = str(vid) + '_' + randid

            # Inserting the new data
            data_ours['metadata'][new_key] = new_value

        if vid not in modified_videos:
            modified_videos.append(vid)



    for i in ["1","2","3","4"]:
        data_ours['config']['file']['loc_prefix'][i] = "file:///E:/tfm/actions/videos" #"file:///%DIR_TO_VIDEOS%"  #TODO

    if verbose: print("Dumping data...")
    with codecs.open(outfile, 'w', encoding='utf-8') as out:
        json.dump(data_ours, out, indent=None, ensure_ascii=False)

    print("All ok!")
    print("Modified videos:", modified_videos)

    # Closing file
    out.close()

if __name__ == "__main__":
    main()