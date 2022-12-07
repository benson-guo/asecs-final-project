import argparse
import codecs
import csv
from ogb.nodeproppred import NodePropPredDataset

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

magToNodeIdx = {}
magToField = {}
paperInfo = {}

# https://learn.microsoft.com/en-us/academic-services/graph/reference-data-schema#papers
class Paper:
    def __init__(self, data, field):
        assert len(data) == 26
        self.id = data[0]
        self.rank = data[1]
        self.docType = data[3]
        self.title = data[4]
        self.originalTitle = data[5]
        self.year = data[7]
        self.date = data[8]
        self.citations = data[19]
        self.estimatedCitations = data[20]
        self.field = field

    def __str__(self):
        return f"Paper(ID: {self.id}, Type: {self.docType}, Field: {self.field}, Title: {self.title}, Date: {self.date}, Citations: {self.citations})"
       

def fix_nulls(s):
    for line in s:
        yield line.replace('\0', ' ')

def parse_data(data_dir):
    dataset = NodePropPredDataset(name = "ogbn-arxiv")
    graph, label = dataset[0]

    with open(f"{data_dir}/arxiv/mapping/nodeidx2paperid.csv", 'r') as fd:
        for idx, row in enumerate(fd):
            if idx == 0:
                # skip header
                continue
            nodeIdx, mag = row.rstrip("\n").split(",")
            #print(row)
            magToNodeIdx[mag] = nodeIdx
            fieldIdx = str(label[int(nodeIdx)].item())
            field = field_mapping[fieldIdx]
            magToField[mag] = field
    print(len(magToNodeIdx))
    with open(f"{data_dir}/papers.txt", 'r') as paperFile:
        for idx, line in enumerate(paperFile):
            line = line.rstrip("\n").split("\t")
            paperId = line[0]
            if paperId not in magToField:
                # not part of ogbn-arxiv
                continue
            print(line)
            field = magToField[paperId]
            paper = Paper(line, field)
            paperInfo[paper.id] = paper
            print(paper)
            print(graph['node_year'][int(magToNodeIdx[paperId])].item())
            import pdb
            pdb.set_trace()
            if idx > 10000:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir")
    args = parser.parse_args()
    parse_data(args.data_dir)
