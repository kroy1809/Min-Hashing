import os
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from scipy.sparse import csr_matrix


class GenerateFeatures:
    def __init__(self, directory):
        self.ngram2ID = {}
        self.nextID = 0
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

        files = os.listdir("%s" % directory)
        if "readme.txt" in files: files.remove("readme.txt")

        row_cnt = 0
        matrix_rows = []
        matrix_cols = []
        matrix_data = []
        created_data = set()

        print("Generating Shingles ...")

        for i in range(len(files)):
            if files[i].endswith(".txt"):
                f = open("%s/%s" % (directory, files[i]))
                for lines in f.readlines():
                    ng_list = lines.split("\t")[0].lower()
                    for char in ng_list:
                        if char in punctuations:
                            ng_list = ng_list.replace(char, "")
                    ngramlist = self.gen_ngrams(ng_list, 3)
                    for ng in ngramlist:
                        if (row_cnt, self.get_ngramID(ng)) not in created_data:
                            matrix_cols.append(self.get_ngramID(ng))
                            matrix_rows.append(row_cnt)
                            matrix_data.append(1)
                        created_data.add((row_cnt, self.get_ngramID(ng)))
                    row_cnt += 1

        self.data_matrix = csr_matrix((matrix_data, (matrix_rows, matrix_cols)),
                                      shape=(max(matrix_rows) + 1, self.nextID))

    def gen_ngrams(self, s, n):
        stemmer = SnowballStemmer(language='english')
        tokens = word_tokenize(s)
        tokens_stem = [stemmer.stem(token) for token in tokens]
        if len(tokens_stem) < n:
            return [" ".join(tokens_stem)]
        ngrams = zip(*[tokens_stem[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def get_ngramID(self, ngram):
        if ngram not in self.ngram2ID:
            self.ngram2ID[ngram] = self.nextID
            self.nextID += 1
        return self.ngram2ID[ngram]

if __name__=="__main__":
    obj = GenerateFeatures("sentiment labelled sentences")
