import pickle
import argparse
import numpy as np

from parse_mag import Paper, PaperAuthor, Affiliation

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


def citation_matrix_institutions(data):
    papers = data["papers"]
    affiliations = data["affiliations"]

    num_affiliations = len(affiliations)
    idx_to_affiliation = {}
    affiliation_to_idx = {}
    for idx, affiliation_id in enumerate(affiliations.keys()):
        idx_to_affiliation[idx] = affiliation_id
        affiliation_to_idx[affiliation_id] = idx

    # citations[x][y] is the number of times affiliation x cites affiliation y
    citations = np.zeros((num_affiliations, num_affiliations))
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
    return citations


def citation_matrix_fields(data):
    papers = data["papers"]
    field_mapping = data["fields"]

    num_fields = len(field_mapping)
    field_to_idx = {v: k for k, v in enumerate(field_mapping.values())}

    # citations[x][y] is the number of times field x cites field y
    citations = np.zeros((num_fields, num_fields))
    for paper in papers.values():
        f1 = field_to_idx[paper.field]
        for referred_paper in paper.referred_papers:
            f2 = field_to_idx[referred_paper.field]
            citations[f1][f2] += 1

    import pdb
    pdb.set_trace()

    return citations


def citations_between_institutions(data, top: int = 100):
    citations = citation_matrix_institutions(data)

    # find most cited institutions
    affiliations = list(data["affiliations"].values())
    num_citations = citations.sum(axis=0).astype(int)
    # argsort in descending order
    sort_id = np.argsort(-num_citations)

    print(f"Top {top} Institutions")
    for idx in sort_id[:top]:
        print(f"Citations: {num_citations[idx]}, Insitution: {affiliations[idx].name}")


def citations_between_fields(data, top: int = 100):
    # find most common inter field citations
    citations = citation_matrix_fields(data)
    idx_to_field = {k: v for k, v in enumerate(data["fields"].values())}
    i, j = [x.reshape(-1) for x in np.indices(citations.shape)]
    citations = citations.reshape(-1).astype(int)

    # keep the inter-field citations
    inter_idx = np.where(i != j)[0]
    citations, i, j = citations[inter_idx], i[inter_idx], j[inter_idx]
    top_citation_patterns = list(zip(*[citations, i, j]))
    top_citation_patterns.sort(reverse=True)

    print("Most common inter field citations")
    for citations, f1, f2 in top_citation_patterns[:top]:
        fn1 = idx_to_field[f1]
        fn2 = idx_to_field[f2]
        print(
            f"Citations: {int(citations)}, {fn1} ({full_field_name[fn1]}) -> {fn2} ({full_field_name[fn2]})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # shared parameters
    parser.add_argument(
        "-d",
        "--data_dir",
        default="data",
        help="Directory where data.pkl resides",
    )
    args = parser.parse_args()
    with open(f"{args.data_dir}/data.pkl", "rb") as fh:
        data = pickle.load(fh)
    citations_between_fields(data)
    citations_between_institutions(data)
