import sys
import zipfile
import json
import re
import os
import heapq
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
from collections import Counter, defaultdict
import warnings
from nltk.stem import PorterStemmer
import time
from search_utils import tokenize

start_time = time.perf_counter()

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

PATH = 'analyst.zip' # switch to analyst for debugging developer
TAG_WHITELIST = ['title','h1','h2','h3', 'h4', 'h5', 'h6','b','strong']
WEIGHTS = {'title': 4,'h1': 2, 'h2': 2, 'h3': 2,'h4': 1, 'h5': 1, 'h6': 1, 'b': 1, 'strong': 1 }
MAX_INDEX_SIZE = 10 * 1024 * 1024 # 10 mb

def open_file(zf, path):
    '''
    Opens the file on path and counts the tokens, returning the url and token count.
    '''
    with zf.open(path, 'r') as f:
        data = json.load(f)
    url = data['url']

    soup = BeautifulSoup(data['content'], 'lxml')
    count = Counter()
    total = 0
    main_content = soup.find("body") # Skip headers
    if not main_content: # Skip empty content
        return url, count, total

    for tag in main_content.find_all(True):
        weight = WEIGHTS.get(tag.name, 1)
        if weight == 1:
            continue  # handled by the flat pass below; skip redundant work
        for string in tag.strings:
            tag_tokens = tokenize(string)
            for token in tag_tokens:
                count[token] += weight

    tokens = tokenize(main_content.get_text(separator = ' ', strip = True))

    for token in tokens:
        count[token] += 1
        total += 1
    return url, count, total

def dump_json(data, dest, partial=False):
    """
    Dumps the dictionary into the destination file.
    """
    print(f"Dumping {dest}...")
    dumping_pre_t = time.perf_counter()
    with open(dest, "w") as f:
        if partial:
            for token in sorted(data.keys()):
                f.write(json.dumps({token: data[token]}) + "\n")
        else:
            json.dump(data, f)
    print(f"Dumping finished. {time.perf_counter() - dumping_pre_t:.4f} seconds.")

def iter_index(path):
    """ Yields from a partial index to avoid loading entire partial index."""
    with open(path, 'r') as f:
        for line in f:
            d = json.loads(line)
            token = next(iter(d))
            yield token, d[token]

def merge_indices(num_of_indices, dest='index.json'):
    """ Merges the partial indices to a single index. """
    print("Merging partial indices...")
    start = time.perf_counter()
    iterators = [iter_index(f"index{i}.json") for i in range(num_of_indices)]
    heap = []

    for i, it in enumerate(iterators):
        try:
            token, postings = next(it)
            heapq.heappush(heap, (token, i, postings))
        except StopIteration:
            pass

    with open(dest, 'w') as f:
        while heap:
            current_token, i, postings = heapq.heappop(heap)
            merged = list(postings)

            while heap and heap[0][0] == current_token: # look at other iterators for same token
                _, j, more = heapq.heappop(heap)
                merged.extend(more)
                try:
                    token, p = next(iterators[j])
                    heapq.heappush(heap, (token, j, p))
                except StopIteration:
                    pass

            merged.sort(key=lambda x: x[0], reverse=True)
            f.write(json.dumps({current_token: merged}) + '\n')

            try:
                token, p = next(iterators[i])
                heapq.heappush(heap, (token, i, p))
            except StopIteration:
                pass
    print(f"Merging finished. {time.perf_counter() - start:.4f} seconds.")

def main():
    stemmer = PorterStemmer()

    page_id = 0
    indices = 0
    unique_tokens = set() # REMOVE AFTER M1
    r_index = defaultdict(list) # swapped to max-heap for more efficient retrieval of top k results, format: [stemmed token: list[(-count, doc id)]]
    id_mapping = {}
    with zipfile.ZipFile(PATH, "r") as zf:
        for filename in zf.namelist():
            if filename.endswith(".json"):
                if __debug__: debug_pre_t = time.perf_counter()
                url, word_count, total = open_file(zf, filename)
                if __debug__: print(filename, len(word_count), total, f"{time.perf_counter()-debug_pre_t:.2f}")
                for key in word_count.keys():
                    r_index[stemmer.stem(key)].append((page_id, word_count[key])) # format: [doc id: list[postings, count]]
                    unique_tokens.add(stemmer.stem(key))
                id_mapping[page_id] = url
                page_id += 1

                if sys.getsizeof(r_index) > MAX_INDEX_SIZE:
                    print("Max index size reached, dumping to disk...")
                    dump_json(r_index, f"index{indices}.json", partial=True)
                    r_index.clear()
                    indices += 1

            elif __debug__:
                print(filename)

    dump_json(r_index, f"index{indices}.json", partial=True)

    merge_indices(indices+1)

    dump_json(id_mapping, "id_mapping.json")

    total_size = os.path.getsize("index.json") / 1024 #get in bytes, then convert to KB
    end_time = time.perf_counter()

    delta_min = (end_time - start_time) // 60
    delta_sec = (end_time - start_time) % 60
    print("Number of partial indices made:", indices+1)
    print("Removing partial indices...")
    for p in [f"index{i}.json" for i in range(indices+1)]:
        os.remove(p)

    print(f"Execution time: {end_time - start_time:.4f} seconds\n")
    print("Number of indexed documents:", len(id_mapping))
    print("Number of unique tokens after stemming:", len(unique_tokens))
    print("Size in KB:", f"{total_size:.2f}")

    # Output.txt
    temp = "output.txt.tmp"
    fin = "output.txt"

    with open(temp, "w") as outFile:
        outFile.write(f"Execution time: {delta_min} minutes, {delta_sec:.2f} seconds\n")
        outFile.write(f"Number of partial indices made: {indices + 1}\n\n")

        outFile.write(f"Number of indexed documents: {len(id_mapping)}\n")
        outFile.write(f"Number of unique tokens after stemming: {len(unique_tokens)}\n")
        outFile.write(f"Size in KB: {total_size:.2f}\n\n")

    os.replace(temp, fin)

if __name__ == '__main__':
    main()