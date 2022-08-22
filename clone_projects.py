import os

import pandas as pd
from tqdm import tqdm

apache = pd.read_csv("./apache.csv")
mit = pd.read_csv("./mit.csv")

projects = pd.concat([apache, mit], ignore_index=True)
projects.sort_values(by="stargazers", ascending=False, inplace=True)

projects_number = 1000
for _, project in tqdm(projects.head(projects_number).iterrows(), total=projects_number):
    project_dir = os.path.join("../datasets/python-1000/", project['name'].replace('/', '__'))
    os.system(f"git clone https://github.com/{project['name']}.git {project_dir}")

    # traverse root directory and remove files without ".py" extension
    for root, dirs, files in os.walk(project_dir, topdown=False):
        for file in files:
            if not file.endswith(".py"):
                os.remove(os.path.join(root, file))
        if not os.listdir(root):
            os.rmdir(root)
