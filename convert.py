import json
import os
import pandas as pd

csv_path = "./input/DataConvert.csv"


def read(file_path):
    if not os.path.exists(file_path):
        print("File Not Found!")
        return
    with open(file_path, encoding='utf-8') as f:
        content = json.loads(f.read())

    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.close()
    init = pd.DataFrame()
    n = 0
    for i in content:
        print(f"Converting：{n}")
        n += 1
        publications = i["publications"]
        titles = []
        for one in publications:
            title = one['title']
            titles.append(title)
        convert = pd.DataFrame({'title': titles})
        init = pd.concat([init, convert], axis=0, ignore_index=True, join='outer')
    print(init)
    init.to_csv(csv_path, index=False, sep=',')


if __name__ == '__main__':
    # you need to modify the file path. if you put the data in ./input/, you don't need something
    path = r'./input/SciKG_min_1.0.txt'
    read(path)
