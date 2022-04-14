from math import log
from typing import TYPE_CHECKING, Iterable, Iterator, List, Sequence, Set, TextIO, TypeVar

import numpy as np
from genutility.file import copen
from genutility.gensim import KeyedVectors
from genutility.nlp import load_freqs
from genutility.scipy import linear_sum_assignment_cost
from networkx import from_scipy_sparse_matrix
from networkx.algorithms.clique import find_cliques
from scipy.sparse import dok_matrix

from .clustering import get_sequential_clusters

if TYPE_CHECKING:

    from spacy.tokens import Doc, Span, Token

T = TypeVar("T")


def avg_norm(x: float, a: float, b: float) -> float:

    return ((x / a) + (x / b)) / 2.0


def isContent(tok: "Token"):
    return tok.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "NUM", "X", "SYM", "INTJ"}


def load_lines(stream: TextIO) -> Iterator[str]:

    for line in stream:
        yield line.rstrip("\n")


class InformationContent:
    def __init__(self, path: str, norm: float = 1.0) -> None:

        self.norm = norm

        self.frequencies = load_freqs(path, normalize=False)

        self.sum_freq = sum(x / norm for x in self.frequencies.values())
        self.min_freq = min(x / norm for x in self.frequencies.values())

        if self.min_freq == 0:
            self.min_freq = 1

    def getInformationContent(self, word: str) -> float:

        num = self.frequencies.get(word, 0) + self.min_freq
        denom = self.norm * self.sum_freq

        return -log(num / denom)

    def getRelativeInformationContent(self, word: str) -> float:

        try:
            num = self.frequencies[word] + self.min_freq
            denom = self.norm * self.sum_freq
            infCont = -log(num / denom)

            maxInfCont = -log(self.min_freq / denom)
            return infCont / maxInfCont
        except KeyError:
            return 1.0


class WordVectorSpace:
    def __init__(
        self,
        path: str,
        archive_file: str = "glove.6B.200d.txt",
        limit: int = 400000,
        vector_size: int = 200,
    ) -> None:

        with copen(path, "rt", encoding="utf-8", archive_file=archive_file) as fr:
            self.embeddings = KeyedVectors.load_glove_format(fr, limit, vector_size)

    def similarity(self, word1: str, word2: str) -> float:

        if word1 == word2:
            return 1.0

        return self.embeddings.similarity(word1, word2)


def word_dissimilarity_matrix(
    tokens_a: "Span", tokens_b: "Span", ic: InformationContent, wvs: WordVectorSpace
) -> np.ndarray:

    maxlen = max(len(tokens_a), len(tokens_b))
    dissimilarities = np.ones((maxlen, maxlen), dtype=np.float)

    for k, token_a in enumerate(tokens_a):
        for l, token_b in enumerate(tokens_b):
            # for all combinations of tokens k, l

            try:
                simTokens = wvs.similarity(token_a.lemma_, token_b.lemma_)
                ic_a = ic.getRelativeInformationContent(token_a.lemma_)
                ic_b = ic.getRelativeInformationContent(token_b.lemma_)

                dissimilarities[k, l] = 1.0 - max(ic_a, ic_b) * simTokens

            except KeyError:
                pass  # matrix is pre-assigned ones

    return dissimilarities


def sentence_similarity(tokens_a: "Span", tokens_b: "Span", ic: InformationContent, wvs: WordVectorSpace) -> np.ndarray:

    dissimilarities = word_dissimilarity_matrix(tokens_a, tokens_b, ic, wvs)
    matching_cost = linear_sum_assignment_cost(dissimilarities)  # Hungarian/Kuhn-Munkres algorithm
    maxlen = max(len(tokens_a), len(tokens_b))
    return avg_norm(maxlen - matching_cost, len(tokens_a), len(tokens_b))


class GraphBuilder:
    def __init__(self, stopwords: Set[str]) -> None:

        self.stopwords = stopwords
        self.localizationSize = 100

    def filter_tokens(self, tokens: Iterable["Token"]) -> Iterator["Token"]:

        for tok in tokens:
            if isContent(tok) and tok.lemma_ not in self.stopwords:
                yield tok

    def sentence_similarity_matrix(
        self, snippets: List["Span"], ic: InformationContent, wvs: WordVectorSpace
    ) -> dok_matrix:

        similarity_matrix = dok_matrix((len(snippets), len(snippets)), dtype=np.float)

        for i in range(0, len(snippets) - 1):
            for j in range(i + 1, min(len(snippets), i + self.localizationSize)):
                # for all combinations of snippets i, j

                tokens_a = list(self.filter_tokens(snippets[i]))
                tokens_b = list(self.filter_tokens(snippets[j]))

                if tokens_a and tokens_b:
                    similarity_matrix[i, j] = sentence_similarity(tokens_a, tokens_b, ic, wvs)

        return similarity_matrix


class GraphSeg:
    def __init__(
        self,
        path_freqs: str,
        path_wordvecs: str,
        path_stop: str,
        treshold: float = 0.25,
        minseg: int = 1,
    ) -> None:

        if treshold < 0 or treshold > 1:
            raise ValueError()

        if minseg < 1:
            raise ValueError()

        self.treshold = treshold
        self.minseg = minseg

        self.ic = InformationContent(path_freqs, 1)
        self.wvs = WordVectorSpace(path_wordvecs)

        with copen(path_stop, "rt") as fr:
            stopwords = set(load_lines(fr))

        self.graphbuilder = GraphBuilder(stopwords)

    @staticmethod
    def concat(snippets: Sequence[T], segmentation: Sequence[Sequence[int]]) -> Iterable[List[T]]:

        for seg in segmentation:
            yield [snippets[ind] for ind in seg]

    def segment(self, doc: "Doc") -> Iterable[List["Span"]]:

        snippets = list(doc.sents)

        similarity_matrix = self.graphbuilder.sentence_similarity_matrix(snippets, self.ic, self.wvs)
        graph = from_scipy_sparse_matrix(similarity_matrix > self.treshold)
        cliques = list(find_cliques(graph))  # Bron-Kerbosch algorithm

        if False:
            from itertools import combinations

            import matplotlib.pyplot as plt
            import networkx as nx
            from genutility.networkx import connected_subgraph

            for clique in cliques:
                if len(clique) > 1:
                    for x, y in combinations(clique, 2):
                        graph.edges[x, y]["color"] = "red"

            graph = connected_subgraph(graph)
            edge_color = [graph.edges[x, y].get("color", "blue") for x, y in graph.edges]

            nx.draw(graph, with_labels=True, edge_color=edge_color)
            plt.show()

        sequential_clusters = get_sequential_clusters(cliques, similarity_matrix, self.minseg)

        return self.concat(snippets, sequential_clusters)
