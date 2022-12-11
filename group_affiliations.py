import pickle
import argparse

import numpy as np
from parse_mag import Paper, PaperAuthor, Affiliation
from analysis import citation_matrix_fields, citation_matrix_institutions

hardcoded_entries = {
    1: [23549],
    2: [5688, 7684, 7947, 14055, 16527, 20400, 20650, 21033],
    3: [
        2267,
        1221,
        3815,
        3821,
        6596,
        6844,
        7954,
        11925,
        12939,
        14202,
        14814,
        15898,
        16770,
        17085,
        17801,
        18587,
        19919,
        22933,
        23195,
    ],
}

mapping_entities = {1: "university", 2: "institute", 3: "company"}


def define_type(affiliation_name, affiliation_type=None):
    if isinstance(affiliation_type, int):
        return affiliation_type

    for word in ["universi", "college", "école", "education"]:
        if word in affiliation_name.lower():
            return 1

    for word in ["institute", "agency", "academy", "hospital", "association"]:
        if word in affiliation_name.lower():
            return 2

    if "company" in affiliation_name.lower():
        return 3

    if affiliation_type:
        for word in ["university", "college", "école", "education", "school"]:
            if word in affiliation_type.lower():
                return 1

        for word in [
            "research",
            "institute",
            "laboratory",
            "academy",
            "agency",
            "profit",
            "hospital",
            "association",
        ]:
            if word in affiliation_type.lower():
                return 2

        for word in ["company", "corporation", "manufacturer", "provider"]:
            if word in affiliation_type.lower():
                return 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # shared parameters
    parser.add_argument(
        "-d",
        "--data_dir",
        default="data",
        help="Directory where data.pkl and affiliation_type_raw.pkl resides",
    )
    args = parser.parse_args()
    with open(f"{args.data_dir}/data.pkl", "rb") as f:
        data = pickle.load(f)

    with open(f"{args.data_dir}/affiliation_type_raw.pkl", "rb") as f:
        file = pickle.load(f)
        idx = np.array(list(file.keys())).astype(int)
        descriptions = np.array(list(file.values()))

    affiliation_group = []
    affiliations = [
        x.name for i, x in enumerate(data["affiliations"].values()) if i in idx
    ]
    for i in range(len(idx)):
        affiliation_group.append(define_type(affiliations[i], descriptions[i]))
    affiliation_group = np.array(affiliation_group)

    # input entries that have been hardcoded (e.g. Intel --> 3)
    for k, v in hardcoded_entries.items():
        affiliation_group[np.where(np.isin(idx, v))[0]] = k

    for k, v in mapping_entities.items():
        print(f"{v.title()} in the top {len(idx)}: {(affiliation_group == k).sum()}")
