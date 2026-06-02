import sys
import zipfile
import json
import os
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
from collections import defaultdict
import warnings
from nltk.stem import PorterStemmer
import time
import search_utils
from search_indexer import dump_json, merge_indices, build_offset_index, MAX_INDEX_SIZE, SIM_REMOVAL, PATH
start_time = time.perf_counter()

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def open_file(zf, path):
    '''
    Opens the file on path and records the token positions, returning the url and token count.
    '''
    stemmer = PorterStemmer()
    with zf.open(path, 'r') as f:
        data = json.load(f)
    url = data['url']

    soup = BeautifulSoup(data['content'], 'lxml')
    positions = defaultdict(list)
    total = 0
    main_content = soup.find("body") # Skip headers
    if not main_content: # Skip empty content
        return url, positions, total

    # no tags or anchors, for simplicity

    tokens = search_utils.tokenize(main_content.get_text(separator = ' ', strip = True))

    for i, token in enumerate(tokens):
        positions[stemmer.stem(token)].append(i)
        total += 1
    return url, positions, total

def main():
    page_id = 0
    indices = 0
    unique_tokens = set() # may delete later
    r_index = defaultdict(list) # swapped to max-heap for more efficient retrieval of top k results, format: [stemmed token: list[(-count, doc id)]]
    id_mapping = {}
    with zipfile.ZipFile(PATH, "r") as zf:
        for filename in zf.namelist():
            if not filename.endswith(".json"): continue

            if __debug__:
                debug_pre_t = time.perf_counter()
                print(filename, end=' ')
            url, token_positions, total = open_file(zf, filename)
            if __debug__:
                print(len(token_positions), total, f"{time.perf_counter()-debug_pre_t:.2f}")
            if SIM_REMOVAL and search_utils.is_similar({token: len(positions) for token, positions in token_positions.items()}): # Simhash 95% remove similar pages
                    continue
            for key in token_positions.keys():
                r_index[key].append((page_id, token_positions[key])) # format: [doc id: list[postings, count]]
                unique_tokens.add(key)
            id_mapping[url] = page_id
            page_id += 1

            if sys.getsizeof(r_index) > MAX_INDEX_SIZE:
                print("Max index size reached, dumping to disk...")
                dump_json(r_index, f"positional_index{indices}.json", partial=True)
                r_index.clear()
                indices += 1

            elif __debug__:
                print(filename)

    dump_json(r_index, f"positional_index{indices}.json", partial=True)

    merge_indices(indices+1, "positional_index")

    dump_json(id_mapping, "positional_index_inv_id_mapping.json")

    total_size = os.path.getsize("positional_index.json") / 1024 #get in bytes, then convert to KB
    end_time = time.perf_counter()

    delta_min = (end_time - start_time) // 60
    delta_sec = (end_time - start_time) % 60
    print("Number of partial indices made:", indices+1)
    print("Removing partial indices...")
    for p in [f"positional_index{i}.json" for i in range(indices+1)]:
        os.remove(p)
    build_offset_index("positional_index.json", "positional_index_offsets.json")

    print(f"Execution time: {end_time - start_time:.4f} seconds\n")
    print("Number of indexed documents:", len(id_mapping))
    print("Number of unique tokens after stemming:", len(unique_tokens))
    print("Size in KB:", f"{total_size:.2f}")

    # Output.txt
    temp = "positional_index_output.txt.tmp"
    fin = "positional_index_output.txt"

    with open(temp, "w") as outFile:
        outFile.write(f"Execution time: {delta_min} minutes, {delta_sec:.2f} seconds\n")
        outFile.write(f"Number of partial indices made: {indices + 1}\n\n")

        outFile.write(f"Number of indexed documents: {len(id_mapping)}\n")
        outFile.write(f"Number of unique tokens after stemming: {len(unique_tokens)}\n")
        outFile.write(f"Size in KB: {total_size:.2f}\n\n")
    os.replace(temp, fin)

if __name__ == '__main__':
    main()