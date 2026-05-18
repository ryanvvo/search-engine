import json
from nltk.stem import PorterStemmer
from search_utils import tokenize
import time

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


def query_search(query, index, id_mapping):
    stemmer = PorterStemmer()

    tokens = [stemmer.stem(token) for token in tokenize(query)]
    if not tokens:
        return []

    all_tokens = []
    for token in tokens:
        if token not in index:
            return []  # will not have anything, as we are using AND to match
        all_tokens.append(index[token])

    doc = []  # list of dictionaries w/ doc ids as keys and scores as values
    for tk in all_tokens:
        matches = {}

        for id, count in tk:
            matches[int(id)] = count
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

def print_results(query, results):
    """
    Prints the results of a query.
    """
    print(f"Top 5 results for '{query}'")
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

def run_retrieval(index_path="index.json", mapping_path="id_mapping.json"):
    log = {}
    print("Retrieving index...")
    start = time.perf_counter()
    index = retrieve(index_path)
    print(f"Retrieving done. {(time.perf_counter() - start):.2f} seconds.")
    with open(mapping_path, "r") as inFile:
        id_mapping = json.load(inFile)
    #1 – cristina lopes
    #2 - machine learning
    #3 - ACM
    #4 - master of software engineering
    while True:
        query = prompt_user()
        if not query:
            print("Exiting...")
            break
        results = query_search(query, index, id_mapping)
        print_results(query, results)
        log[query] = results

    log_output(log)

def main():
    run_retrieval()



if __name__ == '__main__':
    main()