import os

import pandas as pd
from tqdm import tqdm

apache = pd.read_csv("./apache.csv")
mit = pd.read_csv("./mit.csv")

projects = pd.concat([apache, mit], ignore_index=True)
projects.sort_values(by="stargazers", ascending=False, inplace=True)

projects_number = 1000
for _, project in tqdm(projects.head(projects_number).iterrows(), total=projects_number):
    os.system(f"git clone https://github.com/{project['name']}.git "
              f"../datasets/python-1000/{project['name'].replace('/', '__')}/")
