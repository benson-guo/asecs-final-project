import argparse
import pickle

import numpy as np
from group_affiliations import get_affiliation_groups, mapping_entities
from parse_mag import Paper, PaperAuthor, Affiliation

def author_group_name(author_group):
    name = "("
    for i, ag_idx in enumerate(author_group):
        if i > 0:
            name += "+"
        name += mapping_entities[ag_idx]
    name += ")"
    return name
        

def inter_group_citations(data, affiliation_idx, affiliation_group):
    # filter for papers with affiliations that we have grouping information for
    papers = data['papers']
    affiliations = data['affiliations']
    idx_to_afg = {}
    idx_to_id = {}
    id_to_idx = {}
    filtered_papers = []
    author_groups = [(1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)] # different combinations of papers authors
    author_group_to_idx = {}
    for idx, ag in enumerate(author_groups):
        author_group_to_idx[ag] = idx
    num_groups = len(author_groups)
    for idx, id in enumerate(affiliations.keys()):
        idx_to_id[idx] = id
        id_to_idx[id] = idx
    for idx, ag in zip(affiliation_idx, affiliation_group):
        idx_to_afg[idx] = ag

    papers_analyzed = 0
    paper_author_group = {}
    for paper in papers.values():
        author_group = set()
        author_aids = [author.affiliation_id for author in paper.authors]
        skip_paper = any(id not in id_to_idx or id_to_idx[id] not in idx_to_afg for id in author_aids)
        if skip_paper:
            # don't have sufficient affiliation info for one of the authors
            continue
        papers_analyzed += 1
        for aid in author_aids:
            author_group.add(idx_to_afg[id_to_idx[aid]])

        paper_author_group[paper.id] = tuple(author_group)

    citations = np.zeros((num_groups, num_groups)) # how many times one group cites another
    author_group_count = np.zeros(num_groups) # how many of each author group
    for paper in papers.values():
        if paper.id not in paper_author_group:
            continue
        citing_author_group = paper_author_group[paper.id]
        group1 = author_group_to_idx[citing_author_group]
        author_group_count[group1] += 1
        for referred_paper in paper.referred_papers:
            if referred_paper.id not in paper_author_group:
                continue
            cited_author_group = paper_author_group[referred_paper.id]
            group2 = author_group_to_idx[cited_author_group]
            citations[group1][group2] += 1

    for i in range(num_groups):
        total_citations = np.sum(citations[i, :])
        if total_citations == 0:
            continue
        citing_group = author_groups[i]
        print("=============")
        print(f"{author_group_name(citing_group)}, Paper Count: {author_group_count[i]}, Proportion: {100 * author_group_count[i] / np.sum(author_group_count)}%")
        for j in range(num_groups):
            cited_group = author_groups[j]
            if citations[i][j] == 0:
                continue
            print(f"{author_group_name(citing_group)} -> {author_group_name(cited_group)}, Count: {citations[i][j]}, Proportion: {100* citations[i][j] / total_citations}%")
            


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
    with open(f"{args.data_dir}/data.pkl", "rb") as fh:
        data = pickle.load(fh)
    idx, _, affiliation_group = get_affiliation_groups(args.data_dir)
    inter_group_citations(data, idx, affiliation_group)
