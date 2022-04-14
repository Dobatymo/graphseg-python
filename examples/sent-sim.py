import spacy

from graphseg.graphseg import InformationContent, WordVectorSpace, sentence_similarity

if __name__ == "__main__":
    from argparse import ArgumentParser

    DEFAULT_PATH_FREQS = "freqs.txt"
    DEFAULT_PATH_GLOVE = "glove.6B.zip"

    parser = ArgumentParser()
    parser.add_argument("--path-freqs", default=DEFAULT_PATH_FREQS)
    parser.add_argument("--path-glove", default=DEFAULT_PATH_GLOVE)
    args = parser.parse_args()

    ic = InformationContent(args.path_freqs)
    wvs = WordVectorSpace(args.path_glove)

    nlp = spacy.load("en_core_web_sm")

    while True:
        a = input("1>")
        b = input("2>")

        doca = nlp(a)
        docb = nlp(b)
        ss = sentence_similarity(doca, docb, ic, wvs)
        print(ss)
