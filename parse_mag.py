from __future__ import annotations

import sys
import argparse
from typing import List, Dict
import codecs
import csv
from ogb.nodeproppred import NodePropPredDataset
import pickle

field_mapping = {
    "0": "arxiv cs na",
    "1": "arxiv cs mm",
    "2": "arxiv cs lo",
    "3": "arxiv cs cy",
    "4": "arxiv cs cr",
    "5": "arxiv cs dc",
    "6": "arxiv cs hc",
    "7": "arxiv cs ce",
    "8": "arxiv cs ni",
    "9": "arxiv cs cc",
    "10": "arxiv cs ai",
    "11": "arxiv cs ma",
    "12": "arxiv cs gl",
    "13": "arxiv cs ne",
    "14": "arxiv cs sc",
    "15": "arxiv cs ar",
    "16": "arxiv cs cv",
    "17": "arxiv cs gr",
    "18": "arxiv cs et",
    "19": "arxiv cs sy",
    "20": "arxiv cs cg",
    "21": "arxiv cs oh",
    "22": "arxiv cs pl",
    "23": "arxiv cs se",
    "24": "arxiv cs lg",
    "25": "arxiv cs sd",
    "26": "arxiv cs si",
    "27": "arxiv cs ro",
    "28": "arxiv cs it",
    "29": "arxiv cs pf",
    "30": "arxiv cs cl",
    "31": "arxiv cs ir",
    "32": "arxiv cs ms",
    "33": "arxiv cs fl",
    "34": "arxiv cs ds",
    "35": "arxiv cs os",
    "36": "arxiv cs gt",
    "37": "arxiv cs db",
    "38": "arxiv cs dl",
    "39": "arxiv cs dm",
}

class Affiliation:
    def __init__(self, data: List[str]):
        assert len(data) == 14
        self.id = data[0]
        self.rank = data[1]
        self.normalized_name = data[2]
        self.name = data[3]
        self.official_page = data[5]
        self.paper_count = data[7]
        self.paper_family_count = data[8]
        self.citation_count = data[9]
        self.country_code = data[10]
        self.latitude = data[11]
        self.longitude = data[12]

    def __str__(self):
        return f"Affiliation(ID: {self.id}, Name: {self.name}, Papers: {self.paper_count}, Citations: {self.citation_count}, Country: {self.country_code})"


# https://learn.microsoft.com/en-us/academic-services/graph/reference-data-schema#paper-author-affiliations
class PaperAuthor:
    def __init__(self, data: List[str]):
        assert len(data) == 6
        self.paper_id = data[0]
        self.author_id = data[1]
        self.affiliation_id = data[2]
        self.author_sequence_number = data[3]
        self.original_author = data[4]
        self.affiliation = None

    def __str__(self):
        return f"PaperAuthor(Paper: {self.paper_id}, Author: {self.author_id}:{self.original_author}, Affiliation: {self.affiliation_id}:{self.affiliation}, Author #: {self.author_sequence_number})"

    def add_affiliation(self, affiliation: Affiliation):
        self.affiliation = affiliation


# https://learn.microsoft.com/en-us/academic-services/graph/reference-data-schema#papers
class Paper:
    def __init__(self, data: List[str], field: str):
        assert len(data) == 26
        self.id = data[0]
        self.rank = data[1]
        self.doc_type = data[3]
        self.title = data[4]
        self.original_title = data[5]
        self.year = data[7]
        self.date = data[8]
        self.references = data[18]
        self.citations = data[19]
        self.estimated_citations = data[20]
        self.field = field
        self.authors : List[PaperAuthor] = []
        self.referred_papers: List[Paper] = []

    def __str__(self):
        return f"Paper(ID: {self.id}, Type: {self.doc_type}, Field: {self.field}, Title: {self.title}, Date: {self.date}, Authors: {len(self.authors)}, Citations: {self.citations}, References: {self.references})"

    def add_author(self, author: PaperAuthor):
        self.authors.append(author)

    def add_reference(self, paper: Paper):
        self.referred_papers.append(paper)


mag_to_node_idx = {}
mag_to_field = {}
paper_info: Dict[str, Paper] = {}
papers: List[Paper] = []
affiliations: Dict[str, Affiliation] = {}


def parse_data(data_dir):
    sys.setrecursionlimit(50000)
    dataset = NodePropPredDataset(name = "ogbn-arxiv")
    graph, label = dataset[0]

    with open(f"{data_dir}/arxiv/mapping/nodeidx2paperid.csv", 'r') as fd:
        for idx, row in enumerate(fd):
            if idx == 0:
                # skip header
                continue
            node_idx, mag = row.rstrip("\n").split(",")
            #print(row)
            mag_to_node_idx[mag] = node_idx
            field_idx = str(label[int(node_idx)].item())
            field = field_mapping[field_idx]
            mag_to_field[mag] = field
    print(len(mag_to_node_idx))


    # process affiliations
    with open(f"{data_dir}/affiliations.txt", 'r') as a_file:
        for idx, line in enumerate(a_file):
            line = line.rstrip("\n").split("\t")
            affiliation = Affiliation(line)
            affiliations[affiliation.id] = affiliation
            #print(affiliation)
    print("Processed affiliations")


    # process papers
    with open(f"{data_dir}/papers.txt", 'r') as paper_file:
        for idx, line in enumerate(paper_file):
            line = line.rstrip("\n").split("\t")
            paper_id = line[0]
            if paper_id not in mag_to_field:
                # not part of ogbn-arxiv
                continue
            #print(line)
            field = mag_to_field[paper_id]
            paper = Paper(line, field)
            paper_info[paper.id] = paper
            papers.append(paper)
            if len(papers) % 500 == 0:
                print(f"Parsed {len(papers)} papers", flush=True)
                #break
            #print(paper)
            #print(graph['node_year'][int(magToNodeIdx[paperId])].item())


    # process paper authors
    with open(f"{data_dir}/paperAuthorAffilliations.txt", 'r') as a_file:
        for idx, line in enumerate(a_file):
            line = line.rstrip("\n").split("\t")
            author = PaperAuthor(line)
            if author.paper_id not in paper_info:
                continue
            if len(author.affiliation_id) > 0:
                affiliation = affiliations[author.affiliation_id]
                author.add_affiliation(affiliation)
            if idx % 5000 == 0:
                print(f"Parsed {idx} authors")
            #print(author)
            paper_info[author.paper_id].add_author(author)

    # process paper references
    with open(f"{data_dir}/paperReferences.txt", 'r') as a_file:
        for idx, line in enumerate(a_file):
            line = line.rstrip("\n").split("\t")
            paper_id, paper_reference_id = line
            # only keep track of references of papers within obgn-arxiv
            if paper_id not in paper_info or paper_reference_id not in paper_info:
                continue
            paper_info[paper_id].add_reference(paper_info[paper_reference_id])

    print(f"Papers: {len(papers)}")
    data = {"papers": paper_info, "affiliations": affiliations, "fields": field_mapping}
    with open(f"data/data.pkl", "wb") as handle:
        pickle.dump(data, handle)
    print("Done")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir")
    args = parser.parse_args()
    parse_data(args.data_dir)
