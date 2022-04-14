import spacy
from genutility.file import read_file

from graphseg.graphseg import GraphSeg

if __name__ == "__main__":
    from argparse import ArgumentParser

    DEFAULT_MIN_SEG = 3
    DEFAULT_THRESHOLD = 0.25
    DEFAULT_PATH_FREQS = "freqs.txt"
    DEFAULT_PATH_GLOVE = "glove.6B.zip"
    DEFAULT_PATH_STOPWORDS = "stopwords.txt"

    parser = ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--treshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--minseg", type=int, default=DEFAULT_MIN_SEG)
    parser.add_argument("--path-freqs", default=DEFAULT_PATH_FREQS)
    parser.add_argument("--path-glove", default=DEFAULT_PATH_GLOVE)
    parser.add_argument("--path-stopwords", default=DEFAULT_PATH_STOPWORDS)
    args = parser.parse_args()

    graphseg = GraphSeg(
        args.path_freqs,
        args.path_glove,
        args.path_stopwords,
        args.treshold,
        args.minseg,
    )

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(read_file(args.path, "rt", encoding="utf-8"))

    for paragaph in graphseg.segment(doc):
        print(paragaph)
