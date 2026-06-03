import json
import math
from collections import defaultdict
from urllib.parse import urlparse
from nltk.stem import PorterStemmer
from search_utils import tokenize
import time
from pathlib import Path

INDEX_PATH = 'index.json'
MAP_PATH = 'id_mapping.json'
OFFSET_PATH = 'offsets.json'



def index_search(token, offsets, index_path=INDEX_PATH):
    if token not in offsets:
        return []

    with open(index_path, 'rb') as f:
        f.seek(offsets[token])
        return json.loads(f.readline())[token]

def query_search(query, id_mapping, offsets, **kwargs):
    stemmer = PorterStemmer()
    if kwargs.get("two_gram", False):
        tokens = []
        query_tokens = tokenize(query)
        prev = stemmer.stem(next(query_tokens))
        for token in query_tokens:
            token = stemmer.stem(token)
            tokens.append(prev + "-" + token)
            prev = token

    else:
        tokens = [stemmer.stem(token) for token in tokenize(query)]
    # N = total number of documents in the index
    N = len(id_mapping)

    if not tokens:
        return []

    all_tokens = []
    doc_frequencies = {} # Store df(t)

    for token in tokens:
        if kwargs.get("two_gram", False):
            search = index_search(token, offsets, index_path="two_gram_index.json")
        else:
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
        # Track the individual term scores for this specific document
        individual_scores = [scores[doc_id] for scores in doc]
        
        score = sum(individual_scores)

        # Find the highest single term score and the average term score
        max_single_score = max(individual_scores) if individual_scores else 1
        avg_score = base_score / len(individual_scores) if individual_scores else 1
        
        # Balance factor: Penalizes documents if a single rare word represents 
        # an overwhelming percentage of the total score.
        balance_factor = avg_score / max_single_score
        
        # Scale the final score safely
        final_score = base_score * balance_factor
        # --------------------------

        url = id_mapping[str(doc_id)] if str(doc_id) in id_mapping else id_mapping[doc_id]
        ret.append((final_score, url))
        
    return ret
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
    print(f"Top results for '{query}' in {dt:.6f} seconds.")
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


def evaluate_position(query, results, mapping, offsets):
    stemmer = PorterStemmer()
    tokens = {stemmer.stem(token) for token in tokenize(query)}
    positions = {} # token: [id, [positions]]
    for token in tokens:
        postings = index_search(token, offsets, index_path="positional_index.json")
        positions[token] = {page: pos for page, pos in postings}
    for i, (score, url) in enumerate(results):
        page_id = mapping.get(url, -1)
        if page_id == -1: continue

        relevant_positions = [] # list[list[positions in relevant id]]
        for posting_position in positions.values(): # filters positions to only the relevant page
            pos = posting_position.get(page_id)
            if pos:
                relevant_positions.append(pos)
        if len(relevant_positions) < 2: continue

        bonus = 0
        shortest_len = len(min(relevant_positions, key=len))
        for col in range(shortest_len): # naive distance calculating, in order (for speed)
            total_dist = 0
            start = relevant_positions[0][col]
            for row in range(1, len(relevant_positions)):
                total_dist +=  relevant_positions[row][col] - start
            if total_dist == 0:
                continue
            bonus += 1/total_dist

        results[i] = (score + bonus, url)


def filter_and_diversify_results(results, max_per_domain=3):
    """
    Limits the number of results originating from a single domain/folder setup.
    """
    diversified = []
    domain_counts = defaultdict(int)
    
    for score, url in results:
        parsed = urlparse(url)
        # Groups by host domain (e.g., grape.ics.uci.edu)
        domain = parsed.netloc
        
        if domain_counts[domain] < max_per_domain:
            diversified.append((score, url))
            domain_counts[domain] += 1
            
    return diversified

def run_retrieval(index_path=INDEX_PATH, mapping_path=MAP_PATH, offset_path=OFFSET_PATH):
    log = {}

    with open(MAP_PATH, "r") as inFile:
        print("Loading mapping...")
        id_mapping = json.load(inFile)
    with open(OFFSET_PATH, "r") as inFile:
        print("Loading offsets...")
        offsets = json.load(inFile)

    if Path("two_gram_index.json").exists():
        print("Found two-gram index.")
        with open("two_gram_" + MAP_PATH, "r") as inFile:
            print("Loading two-gram mapping...")
            two_gram_mapping = json.load(inFile)
        with open("two_gram_" + OFFSET_PATH, "r") as inFile:
            print("Loading two-gram offsets...")
            two_gram_offsets = json.load(inFile)

    if Path("positional_index.json").exists():
        print("Positional index found.")
        with open("positional_index_inv_" + MAP_PATH, "r") as inFile:
            print("Loading positional-index inverse mapping...")
            positional_inverse_index_mapping = json.load(inFile)
        with open("positional_index_" + OFFSET_PATH, "r") as inFile:
            print("Loading positional-index offsets...")
            positional_index_offsets = json.load(inFile)

    if Path("hits_results.json").exists():
        print("Hits results found...")
        with open("hits_results.json", 'r') as inFile:
            print("Loading hits-results...")
            hits_results = json.load(inFile)

    if Path("pr_results.json").exists():
        print("PR results found...")
        with open("pr_results.json", 'r') as inFile:
            print("Loading pr-results...")
            pr_results = json.load(inFile)

    while True:
        query = prompt_user()

        if not query:
            print("Exiting...")
            break

        start = time.perf_counter()
        results = query_search(query, id_mapping, offsets)

        if Path("two_gram_index.json").exists(): # Add results of two-gram if it exists
            two_gram_results = query_search(query, two_gram_mapping, two_gram_offsets, two_gram=True)
            score_map = defaultdict(float)
            for score, url in results + two_gram_results:
                score_map[url] += score
            results = [(score, url) for url, score in score_map.items()]

        if Path("positional_index.json").exists(): # Add results of positional if it exists
            evaluate_position(query, results, positional_inverse_index_mapping, positional_index_offsets)

        if Path("hits_results.json").exists():
            for i, (score, url) in enumerate(results):
                results[i] = (score + score * hits_results[url]['authority'], url)

        if Path("pr_results.json").exists():
            for i, (score, url) in enumerate(results):
                results[i] = (score + score * pr_results[url], url)

        results.sort(reverse=True, key=lambda x: x[0])
        results = filter_and_diversify_results(results, max_per_domain=3)

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