import pickle
import argparse
from parse_mag import Paper, PaperAuthor, Affiliation
import numpy as np

# detailed descriptions: https://arxiv.org/archive/cs
full_field_name = {
    "arxiv cs na": "Numerical Analysis",
	"arxiv cs mm": "Multimedia",
	"arxiv cs lo": "Logic in Computer Science",
	"arxiv cs cy": "Computers and Society",
	"arxiv cs cr": "Cryptography and Security",
	"arxiv cs dc": "Distributed, Parallel, and Cluster Computing",
	"arxiv cs hc": "Human-Computer Interaction",
	"arxiv cs ce": "Computational Engineering, Finance, and Science",
	"arxiv cs ni": "Networking and Internet Architecture",
	"arxiv cs cc": "Computational Complexity",
	"arxiv cs ai": "Artificial Intelligence",
	"arxiv cs ma": "Multiagent Systems",
	"arxiv cs gl": "General Literature",
	"arxiv cs ne": "Neural and Evolutionary Computing",
	"arxiv cs sc": "Symbolic Computation",
	"arxiv cs ar": "Hardware Architecture",
	"arxiv cs cv": "Computer Vision and Pattern Recognition",
	"arxiv cs gr": "Graphics",
	"arxiv cs et": "Emerging Technologies",
	"arxiv cs sy": "Systems and Control",
	"arxiv cs cg": "Computational Geometry",
	"arxiv cs oh": "Other Computer Science",
	"arxiv cs pl": "Programming Languages",
	"arxiv cs se": "Software Engineering",
	"arxiv cs lg": "Machine Learning",
	"arxiv cs sd": "Sound",
	"arxiv cs si": "Social and Information Networks",
	"arxiv cs ro": "Robotics",
	"arxiv cs it": "Information Theory",
	"arxiv cs pf": "Performance",
	"arxiv cs cl": "Computation and Language",
	"arxiv cs ir": "Information Retrieval",
	"arxiv cs ms": "Mathematical Software",
	"arxiv cs fl": "Formal Languages and Automata Theory",
	"arxiv cs ds": "Data Structures and Algorithms",
	"arxiv cs os": "Operating Systems",
	"arxiv cs gt": "Computer Science and Game Theory",
	"arxiv cs db": "Databases",
	"arxiv cs dl": "Digital Libraries",
	"arxiv cs dm": "Discrete Mathematics",
}

def citations_between_institutions(data):
    papers = data["papers"]
    affiliations = data["affiliations"]

    num_affiliations = len(affiliations)
    idx_to_affiliation = {}
    affiliation_to_idx = {}
    for idx, affiliation_id in enumerate(affiliations.keys()):
        idx_to_affiliation[idx] = affiliation_id
        affiliation_to_idx[affiliation_id] = idx
    # citations[x][y] is the number of times affiliation x cites affiliation y
    citations = [[0 for _ in range(num_affiliations)] for _ in range(num_affiliations)]
    for paper in papers.values():
        paper_affiliations = set()
        for author in paper.authors:
            if author.affiliation is None:
                continue
            paper_affiliations.add(author.affiliation_id)
        for referred_paper in paper.referred_papers:
            reference_affiliations = set()
            for referred_author in referred_paper.authors:
                if referred_author.affiliation is None:
                    continue
                reference_affiliations.add(referred_author.affiliation_id)
            for a1 in paper_affiliations:
                for a2 in reference_affiliations:
                    # affiliation a1 cites a2
                    citations[affiliation_to_idx[a1]][affiliation_to_idx[a2]] += 1
    # find most cited institutions
    citations = np.array(citations)
    top_institutions = []
    for idx in range(num_affiliations):
        citations_received = np.sum(citations[:, idx])
        top_institutions.append((citations_received, idx))
    top_institutions.sort(reverse=True)

    print("Top Institutions")
    for citations_received, idx in top_institutions[:100]:
        affiliation = affiliations[idx_to_affiliation[idx]]
        print(f"Citations: {citations_received}, Insitution: {affiliation.name}")
            

def citations_between_fields(data):
    papers = data["papers"]
    field_mapping = data["fields"]    
    
    num_fields = len(field_mapping)
    idx_to_field = {}
    field_to_idx = {}
    for idx, field in enumerate(field_mapping.values()):
        idx_to_field[idx] = field
        field_to_idx[field] = idx
    # citations[x][y] is the number of times field x cites field y
    citations = [[0 for _ in range(num_fields)] for _ in range(num_fields)]
    for paper in papers.values():
        f1 = field_to_idx[paper.field]
        for referred_paper in paper.referred_papers:
            f2 = field_to_idx[referred_paper.field]
            citations[f1][f2] += 1
    # find most common inter field citations
    citations = np.array(citations)
    top_citation_patterns = []
    for i in range(num_fields):
        for j in range(num_fields):
            top_citation_patterns.append((citations[i][j], i, j))
    top_citation_patterns.sort(reverse=True)
    print("Most common inter field citations")
    for citations, f1, f2 in top_citation_patterns[:100]:
        fn1 = idx_to_field[f1]
        fn2 = idx_to_field[f2]
        print(f"Citations: {citations}, {fn1} ({full_field_name[fn1]}) -> {fn2} ({full_field_name[fn2]})")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # shared parameters
    parser.add_argument(
        "-d",
        "--data_dir",
        default="dataset",
        help="Directory where data.pkl resides",
    )
    args = parser.parse_args()
    with open(f"{args.data_dir}/data.pkl",'rb') as fh:
        data = pickle.load(fh)
    citations_between_fields(data)
    citations_between_institutions(data)
