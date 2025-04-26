from pathlib import Path
import os
import sys
import networkx as nx
from matplotlib import pyplot as plt
import re
from std_headers import std_headers
import difflib


def extract_includes(file):
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        pattern = r'#include\s*<([^>]+)>'
        matches = re.findall(pattern, content)
        return [Path(match) for match in matches]


def is_in_path(subpath, fullpath):
    return str(fullpath).endswith(str(subpath))


def match_paths(subpaths, fullpaths):
    matches = []
    for subpath in subpaths:
        for fullpath in fullpaths:
            if is_in_path(subpath, fullpath):
                matches.append(fullpath)
    return matches


def create_dependency_graph(source_files, header_files):

    dependency_edges = []
    std_includes = []
    for subject_source in source_files:
        includes = extract_includes(subject_source)
        dependencies = match_paths(includes, header_files)
        dependency_edges.extend([(subject_source, header)
                                for header in dependencies])
        std_includes.extend(extract_std_directives(subject_source))
        dependency_edges.extend([(subject_source, header, {"type": "std"})
                                for header in includes])
    for subject_header in header_files:
        includes = extract_includes(subject_header)
        dependencies = match_paths(includes, header_files)
        dependency_edges.extend([(source, subject_header)
                                for source in dependencies])
        std_includes.extend(extract_std_directives(subject_header))
        dependency_edges.extend([(subject_source, header, {"type": "std"})
                                for header in includes])
    std_includes = list(set(std_includes))
    nodes = source_files + header_files + std_includes
    colors = ["blue"] * len(source_files) + ["green"] * len(
        header_files) + ["red"] * len(std_includes)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(dependency_edges)
    return G, colors


if __name__ == '__main__':
    cwd = Path(__file__).parent

    # set cwd
    os.chdir(cwd)
    include_dir = cwd.parents[1] / "include"
    source_dir = cwd.parents[1] / "src"
    header_files = list(set(include_dir.rglob("*.hpp")))
    source_files = list(set(source_dir.rglob("*.cpp")))

    G, node_colors = create_dependency_graph(source_files, header_files)
    nx.draw(G, with_labels=True, node_color=node_colors)
    plt.show()
