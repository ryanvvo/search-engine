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
from search_indexer import dump_json, merge_indices, build_offset_index, MAX_INDEX_SIZE, SIM_REMOVAL, PATH, canonicalize_url, SIZE_OF_TUPLE
from collections import Counter
start_time = time.perf_counter()

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
stemmer = PorterStemmer()

def open_file(zf, path):
    '''
    Opens the file on path and records the token positions, returning the url and token count.
    '''
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
    r_index = defaultdict(list) # swapped to max-heap for more efficient retrieval of top k results, format: [stemmed token: list[(-count, doc id)]]
    id_mapping = {}
    seen_urls = set()
    posting_count = 0

    with zipfile.ZipFile(PATH, "r") as zf:
        for filename in zf.namelist():
            if not filename.endswith(".json"): continue

            if __debug__:
                debug_pre_t = time.perf_counter()
                print(filename, end=' ')
            url, token_positions, total = open_file(zf, filename)

            canonical_url = canonicalize_url(url)   # Transform path
            # Deduplication Check: Skip if an alternate string version has already been indexed
            if canonical_url in seen_urls:
                if __debug__:
                    print(f"dupe skipped -> {url}")
                continue
            seen_urls.add(canonical_url) # Mark this safe canonical layout as indexed

            if __debug__:
                print(len(token_positions), total, f"{time.perf_counter()-debug_pre_t:.2f}")
            word_counts = Counter({token: len(positions) for token, positions in token_positions.items()})
            if SIM_REMOVAL and search_utils.is_similar(word_counts): # Simhash 95% remove similar pages
                continue
            for key in token_positions.keys():
                r_index[key].append((page_id, token_positions[key])) # format: [doc id: list[postings, count]]
                posting_count += len(token_positions[key])
            id_mapping[canonical_url] = page_id
            page_id += 1

            if posting_count * SIZE_OF_TUPLE > MAX_INDEX_SIZE:
                print("Max index size reached, dumping to disk...")
                dump_json(r_index, f"positional_index{indices}.json", partial=True)
                r_index.clear()
                indices += 1
                posting_count = 0

    dump_json(r_index, f"positional_index{indices}.json", partial=True)

    unique_tokens = merge_indices(indices+1, "positional_index")

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
    print("Number of unique tokens after stemming:", unique_tokens)
    print("Size in KB:", f"{total_size:.2f}")

    # Output.txt
    temp = "positional_index_output.txt.tmp"
    fin = "positional_index_output.txt"

    with open(temp, "w") as outFile:
        outFile.write(f"Execution time: {delta_min} minutes, {delta_sec:.2f} seconds\n")
        outFile.write(f"Number of partial indices made: {indices + 1}\n\n")

        outFile.write(f"Number of indexed documents: {len(id_mapping)}\n")
        outFile.write(f"Number of unique tokens after stemming: {unique_tokens}\n")
        outFile.write(f"Size in KB: {total_size:.2f}\n\n")
    os.replace(temp, fin)

if __name__ == '__main__':
    main()