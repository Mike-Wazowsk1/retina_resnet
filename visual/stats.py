import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import os
import argparse

def load_files(name=None, group=None):
    fol = os.listdir('face/')
    counter = {'Angry': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0, 'Sad': 0, 'Surprise': 0, 'Neutral': 0}
    if group:
        for f in fol:
            if group in f:
                counter[list(f.split('_') & counter.keys())[0]] += 1
        return counter
    if name:
        for f in fol:
            if name in f:
                counter[list(f.split('_') & counter.keys())[0]] += 1
        return counter
    if not name and not group:
        for f in fol:
            counter[list(f.split('_') & counter.keys())[0]] += 1
        return counter


def plotti(name=None, group=None):
    colors = sns.color_palette('pastel')[0:5]
    arr = dict()
    if name and group:
        arr = load_files(name=name, group=group)
    if name and group is None:
        arr = load_files(name=name)
    if group and name is None:
        arr = load_files(group=group)
    if not group and not name:
        arr = load_files()

    fig = plt.figure(figsize=(15, 12))
    ax1 = fig.add_subplot(1, 1, 1)
    labels = [k for k, v in arr.items() if v != 0]
    val = [v for k, v in arr.items() if v != 0]
    explode = [0.05 for x in range(len(labels))]
    title = None
    if group is not None:
        title = f'History emotions of group {group}:'
    if name is not None:
        title = f'History emotions of employee {name}:'
    if name is None and group is None:
        title = f'History emotions of all'
    if name is not None and group is not None:
        title = f'History emotions of employee {name} from {group} group'
    plt.title(title)
    ax1.pie(val, labels=labels, shadow=True, startangle=90,
            autopct=lambda p: '{:.0f}%'.format(round(p)) if p > 1 else '', colors=colors, explode=explode)
    ax1.axis('equal')
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("-n", '--name', required=False, type=str, help="name of employee")
parser.add_argument("-g", '--group', required=False, type=str, help="group of workers")
args = parser.parse_args()
name = args.name
group = args.group
if name is not None and group is not None:
    plotti(name, group)

