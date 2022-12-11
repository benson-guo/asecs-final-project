import pickle
import argparse
import requests
import urllib.parse

import numpy as np
from bs4 import BeautifulSoup
from parse_mag import Paper, PaperAuthor, Affiliation
from analysis import citation_matrix_institutions

headers = {
    'User-agent':
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
        "Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582"
}

def get_affiliation_type(query: str):
    query = urllib.parse.quote_plus(query)
    html = requests.get(f'https://www.google.com/search?q={query}&hl=en', headers=headers)
    result = BeautifulSoup(html.text).find_all("div", {'class':"BNeawe tAd8D AP7Wnd"})
    html.close()

    if result:
        return result[0].text
    return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # shared parameters
    parser.add_argument(
        "-d",
        "--data_dir",
        default="data",
        help="Directory where data.pkl resides",
    )
    parser.add_argument("-t", "-top", defaut=100, type=int, help="Number of top affiliations to process")
    args = parser.parse_args()

    with open(f"{args.data_dir}/data.pkl",'rb') as fh:
        data = pickle.load(fh)

    affiliations = np.array([x.name for x in data["affiliations"].values()])
    cit_inst = citation_matrix_institutions(data)
    idx = np.where(cit_inst.sum(axis=0) >= args.top)[0]

    description = list()
    for affiliation in affiliation[idx]:
        description.append(get_affiliation_type(affiliation))
    description = [x.split('\n')[-1].split('·')[0] for x in description]

    with open(f"{args.data_dir}/affiliation_type_raw.pkl", "wb") as f:
        pickle.dump(dict(zip(idx, description)), f)
