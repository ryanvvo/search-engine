import json
import math
from nltk.stem import PorterStemmer
from search_utils import tokenize
import time
from pathlib import Path

INDEX_PATH = 'index.json'
MAP_PATH = 'id_mapping.json'
OFFSET_PATH = 'offsets.json'
def retrieve(path='index.json'):
    """
    Loads and returns inverted index based on provided path.
    """
    index = {}
    with open(path, 'r') as inFile:
        for line in inFile:
            ln = json.loads(line)
            token = next(iter(ln))
            index[token] = ln[token]

    return index

def index_search(token, offsets, index_path=INDEX_PATH):
    if token not in offsets:
        return []

    with open(index_path, 'rb') as f:
        f.seek(offsets[token])
        return json.loads(f.readline())[token]

def query_search(query, id_mapping, offsets):
    stemmer = PorterStemmer()

    # N = total number of documents in the index
    N = len(id_mapping)

    tokens = [stemmer.stem(token) for token in tokenize(query)]
    if not tokens:
        return []

    all_tokens = []
    doc_frequencies = {} # Store df(t)

    for token in tokens:
        search = index_search(token, offsets)
        if not search:
            return []  # will not have anything, as we are using AND to match
        all_tokens.append(search)
        # df(t) is the number of documents containing this token
        doc_frequencies[token] = len(search)

    doc = []  # list of dictionaries w/ doc ids as keys and scores as values
    for tk, token in zip(all_tokens, tokens):
        matches = {}
        df_t = doc_frequencies[token]
        
        # Calculate IDF for this token
        # Adding 1 to avoid potential division by zero if df_t is somehow 0
        idf = math.log(N / (df_t + 1)) 

        for doc_id, count in tk:
            # tf_weight = 1 + math.log(count) if count > 0 else 0
            # Else, tf_weight = count
            tf_weight = 1 + math.log(count) if count > 0 else 0
            
            # Calculate TF-IDF score for this term in this document
            matches[int(doc_id)] = tf_weight * idf
            
        doc.append(matches)

    overlaps = set(doc[0].keys())  # use first element as base
    for scores in doc[1:]:
        overlaps &= set(scores.keys())

    ret = []
    for doc_id in overlaps:
        score = sum(scores[doc_id] for scores in doc)

        url = id_mapping[str(doc_id)] if str(doc_id) in id_mapping else id_mapping[doc_id]
        ret.append((score, url))

    ret.sort(reverse=True, key=lambda x: x[0])

    return ret[:5]  # top 5

def prompt_user():
    """
    Prompts the user to enter a query and returns it.
    """
    query = input("Enter a query: ")
    return query

def print_results(query, results, dt):
    """
    Prints the results of a query.
    """
    print(f"Top 5 results for '{query}' in {dt:.6f} seconds.")
    for rank, (score, url) in enumerate(results, start=1):
        print(f"{rank}. {url}  score={score}")

def log_output(log):
    print("Storing output to retrieval_results.txt...")
    with open("retrieval_results.txt", "w") as outFile:
        for query, results in log.items():
            outFile.write(f"Query: {query}\n")

            if not results:
                outFile.write("No results found.\n\n")

            for rank, (score, url) in enumerate(results, start=1):
                outFile.write(f"{rank}. {url}  score={score}\n")

            outFile.write("\n")

def run_retrieval(index_path=INDEX_PATH, mapping_path=MAP_PATH, offset_path=OFFSET_PATH):
    log = {}
    with open(mapping_path, "r") as inFile:
        id_mapping = json.load(inFile)
    with open(offset_path, "r") as inFile:
        offsets = json.load(inFile)
    #1 – cristina lopes
    #2 - machine learning
    #3 - ACM
    #4 - master of software engineering
    while True:
        query = prompt_user()
        if not query:
            print("Exiting...")
            break
        start = time.perf_counter()
        results = query_search(query, id_mapping, offsets)
        print_results(query, results, time.perf_counter() - start)
        log[query] = results

    log_output(log)

def main():
    if not (Path(INDEX_PATH).exists and Path(MAP_PATH).exists and Path(OFFSET_PATH).exists):
        print("Please run indexer first!")
        return

    run_retrieval()



if __name__ == '__main__':
    main()