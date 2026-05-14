import zipfile
import json
import re
import os
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
from pathlib import Path
from collections import Counter, defaultdict
import warnings
from nltk.stem import PorterStemmer
import time
start_time = time.perf_counter()

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

PATH = 'developer.zip' # switch to analyst for debugging
TAG_WHITELIST = ['title','h1','h2','h3', 'h4', 'h5', 'h6','b','strong']
WEIGHTS = {'title': 4,'h1': 2, 'h2': 2, 'h3': 2,'h4': 1, 'h5': 1, 'h6': 1, 'b': 1, 'strong': 1 }

def tokenize(text):
    '''
    Reads in text file and returns a normalized list.
    a token is a sequence of alphanumeric characters, independent of capitalization (so Apple, apple, aPpLe are the same token).
    returns Generator<Token>
    '''

    return (match.group() for match in re.finditer(r"[a-z0-9]+", text.lower()))

def open_file(path):
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
        weight = WEIGHTS.get(tag.parent.name, 1)
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

page_id = 0
r_index = defaultdict(list) # Probably switch to max-heap later down the road
id_mapping = {}
with zipfile.ZipFile(PATH, "r") as zf:
    for filename in zf.namelist():
        if filename.endswith(".json"):
            if __debug__: debug_pre_t = time.perf_counter()
            url, word_count, total = open_file(filename)
            if __debug__:
                debug_post_t= time.perf_counter()
                filename = Path(filename)
                print(filename.stem, len(word_count), total, f"{debug_post_t-debug_pre_t:.2f}")  # debug
            for key in word_count.keys():
                r_index[key].append((page_id, word_count[key])) # format: [file_name, count_of_word], probably switch to doc id later
            id_mapping[page_id] = url
            page_id += 1
        else:
            print(filename)

print("Dumping index.json...")
dumping_pre_t = time.perf_counter()
with open("index.json", "w") as f:
    json.dump(r_index, f)
print(f"Dumping finished. {time.perf_counter() - dumping_pre_t:.4f} seconds.")
print("Dumping mapping.json...")
dumping_pre_t = time.perf_counter()
with open("mapping.json", "w") as f:
    json.dump(id_mapping, f)
print(f"Dumping finished. {time.perf_counter() - dumping_pre_t:.4f} seconds.")


total_size = os.path.getsize("index.json") / 1024 #get in bytes, then convert to KB
end_time = time.perf_counter()

delta_t = end_time - start_time
delta_min = (end_time - start_time) // 60
delta_sec = (end_time - start_time) % 60

print(f"Execution time: {end_time - start_time:.4f} seconds\n")
print("Number of indexed documents:", len(id_mapping))
print("Number of unique tokens:", len(r_index))
print("Size in KB:", f"{total_size:.2f}")

def update_stats():
    '''
    Writes to output.txt the analytics.
    '''
    temp = "output.txt.tmp"
    fin = "output.txt"

    with open(temp, "w") as outFile:
        outFile.write(f"Execution time: {delta_min} minutes, {delta_sec:.2f} seconds\n\n")
        outFile.write(f"Number of indexed documents: {len(id_mapping)}\n")
        outFile.write(f"Number of unique tokens: {len(r_index)}\n")
        outFile.write(f"Size in KB: {total_size:.2f}\n\n")

    os.replace(temp, fin)
update_stats()