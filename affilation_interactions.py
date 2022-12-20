import argparse
import pickle

import numpy as np
from group_affiliations import get_affiliation_groups, mapping_entities
from analysis import full_field_name
from parse_mag import Paper, PaperAuthor, Affiliation
import matplotlib.pyplot as plt
import networkx as nx

def author_group_name(author_group, newline=False):
    name = ""
    for i, ag_idx in enumerate(author_group):
        if i > 0:
            if newline:
                name += "\n"
            name += "&"
        # name += mapping_entities[ag_idx]
        name += mapping_entities_binary[ag_idx]
    return name

mapping_entities_binary = {1: "Academia", 3: "Industry"}

# copied from: https://stackoverflow.com/a/70245742
def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items

# analyze citations between different affiliation groups
# filter for papers in fields, if list is empty consider all fields
def inter_group_citations(data, affiliation_idx, affiliation_group, fields=[]):
    # group universities with institutes
    affiliation_group = [3 if x == 3 else 1 for x in affiliation_group]

    # filter for papers with affiliations that we have grouping information for
    papers = data['papers']
    affiliations = data['affiliations']
    idx_to_afg = {}
    idx_to_id = {}
    id_to_idx = {}
    filtered_papers = []
    # author_groups = [(1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)] # different combinations of papers authors
    author_groups = [(1,), (3,), (1, 3)]  # different combinations of papers authors
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
        skip_field = len(fields) > 0 and full_field_name[paper.field] not in fields
        if skip_paper or skip_field:
            # don't have sufficient affiliation info for one of the authors
            continue
        papers_analyzed += 1
        for aid in author_aids:
            author_group.add(idx_to_afg[id_to_idx[aid]])

        paper_author_group[paper.id] = tuple(author_group)

    citations = np.zeros((num_groups, num_groups)) # how many times one group cites another
    author_group_count = np.zeros(num_groups) # how many citations of each author group
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
    citation_proportion = citations / np.sum(citations, axis=1).reshape(-1, 1)
    total_citations = np.sum(citations, axis=0)
    for i in range(num_groups):
        if total_citations[i] == 0:
            continue
        citing_group = author_groups[i]
        print("=============")
        print(f"{author_group_name(citing_group)}, Paper Count: {author_group_count[i]}, Proportion: {100 * author_group_count[i] / np.sum(author_group_count)}%"
              f", Total Citations: {total_citations[i]}, Proportion: {100 * total_citations[i] / np.sum(citations)}%")
        for j in range(num_groups):
            cited_group = author_groups[j]
            if citations[i][j] == 0:
                continue
            print(f"{author_group_name(citing_group)} -> {author_group_name(cited_group)}, Count: {citations[i][j]}, Proportion: {100* citation_proportion[i][j]}%")

    # plot
    figsize = 8
    node_color = ["blue", "red", "green"]
    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    edge_weights = 3.0 * citation_proportion
    edge_labels = {}
    # edge_weights = 3.0 * citations / citations.max()
    node_size = 2000 * total_citations / total_citations.max()
    G = nx.MultiDiGraph()
    for i in range(num_groups):
        G.add_node(i)

    for i in range(num_groups):
        for j in range(num_groups):
            G.add_edge(i, j, weight=edge_weights[i][j])
            edge_labels[(i, j)] = f"{100 * citation_proportion[i][j]:.0f}%"

    # prob want to group institute with university or company to make easier to see visualization
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G,
       pos,
        dict(zip(range(num_groups), [author_group_name(author_groups[i], newline=True) for i in range(num_groups)])),
       # node_color=["blue" for _ in range(num_groups)],
       node_color=node_color,
       node_size=node_size,
       alpha=0.4,
    )

    for edge in G.edges(data='weight'):
        # edge_color = colors["edge"][tuple(affiliation_group[top_k_idx[list(edge[:2])]])]
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=edge[2], alpha=1, arrowsize=10, edge_color=node_color[edge[0]], connectionstyle='arc3, rad = 0.1')
    # nx.draw(G, pos, with_labels=True, connectionstyle='arc3, rad = 0.1')
    my_draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_color='black',
        rotate=False,
        rad=0.1,
        font_size=15,
        alpha=1.0,
    )
    for i, ag in enumerate(author_groups):
        ax.plot([0], [0], color=node_color[i], label=author_group_name(ag))

    def nudge(pos, x_shift, y_shift):
        return {n: (x + x_shift, y + y_shift) for n, (x, y) in pos.items()}

    pos_nodes = nudge(pos, 0, -0.13)
    nx.draw_networkx_labels(G, pos=pos_nodes, labels=dict(zip(range(num_groups), [author_group_name(author_groups[i], newline=True) for i in range(num_groups)]))
                            , font_size=14, font_weight="bold")
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig("aff_int_vis.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # shared parameters
    parser.add_argument(
        "-d",
        "--data_dir",
        default="data",
        help="Directory where data.pkl and affiliation_type_raw.pkl resides",
    )
    parser.add_argument("-f", "--fields", nargs="+", default=[])
    args = parser.parse_args()
    with open(f"{args.data_dir}/data.pkl", "rb") as fh:
        data = pickle.load(fh)
    idx, _, affiliation_group = get_affiliation_groups(args.data_dir)
    inter_group_citations(data, idx, affiliation_group, fields=args.fields)
