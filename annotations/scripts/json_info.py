
import json
import sys
import codecs, unicodedata
import csv

import argparse

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=str, help="Path to the json of the dataset")
    parser.add_argument('out_path', nargs='?', default="out.csv", help="Path to the out csv file")
    parser.add_argument("-m","--mode", default='full', help="Info to excract", choices=['full', 'actions', 'noums'])
    parser.add_argument("-sm","--sort_mode", default='frec', help="Sorting mode, alphabetical or frequency", choices=['alph', 'frec'])
    parser.add_argument("-so","--sort_order", default='desc', help="Sorting order (by frequency or alph)", choices=['asc', 'desc'])

    return parser.parse_args()

def main():

    args = parseargs()

    json_path = args.json_path
    out_path = args.out_path
    mode = args.mode
    sort_mode = args.sort_mode

    sort_order = args.sort_order
    reverse_order = False if sort_order == 'asc' else True

    file1 = json_path

    # Opening JSON file
    f1 = open(file1, encoding='utf-8')
    
    # returns JSON object as a dictionary
    data = json.load(f1)

    # Get the full actions
    actions = {}
    roi_count = 0
    full_count = 0
    for tag, value in zip(data['metadata'], data['metadata'].values()):

        if '2' in value['av']:
            roi_count += 1
        else:
            full_count += 1
            action = str(value['av']['1'])

            if action in actions.keys():
                actions[action] += 1
            else:
                actions[action]  = 1
            
    if sort_mode == 'frec':
        actions = dict(sorted(actions.items(), reverse=reverse_order, key=lambda item: item[1]))
    elif sort_mode == 'alph':
        actions = dict(sorted(actions.items(), reverse=reverse_order))

    # Split in actions an noums
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

    if sort_mode == 'frec':
        just_actions = dict(sorted(just_actions.items(), reverse=reverse_order, key=lambda item: item[1]))
        just_nouns = dict(sorted(just_nouns.items(), reverse=reverse_order, key=lambda item: item[1]))
    elif sort_mode == 'alph':
        just_actions = dict(sorted(just_actions.items(), reverse=reverse_order))
        just_nouns = dict(sorted(just_nouns.items(), reverse=reverse_order))


    out_data = None

    if mode == 'full':
        out_data = actions
    elif mode == 'actions':
        out_data = just_actions
    elif mode == 'noums':
        out_data = just_nouns


    frec1 = 0
    with codecs.open(out_path, 'w', encoding='utf-8') as out:

        # create the csv writer
        writer = csv.writer(out)

        for item, times in zip(out_data, out_data.values()):
            if times <= 1:
                frec1 += 1

            #out.write(item + ", " + str(times) + "\n")
            writer.writerow([item ,str(times)])

    #print("Acciones con frecuencia 1: ", frec1)

    # Closing file
    f1.close()
    out.close()

    total_noums = 0
    total_action = 0

    print("ROI Count: ", roi_count)
    print("Full Count: ", full_count)
    print("Total unique count:" , len(actions))
    print("Total unique Actions: ", len(just_actions))
    print("Total unique Nouns: ", len(just_nouns))


    print("Done!")

if __name__ == "__main__":
    main()