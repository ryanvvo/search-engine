import zipfile
import json
import re
import os
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from collections import Counter, defaultdict
import warnings

import time
start_time = time.perf_counter()

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

PATH = 'developer.zip' # switch to analyst for debugging
def tokenize(text):
    '''
    Reads in text file and returns a list of the tokens in that file. For the purposes of this project,
    a token is a sequence of alphanumeric characters, independent of capitalization (so Apple, apple, aPpLe are the same token).
    returns List<Token>
    '''

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    total = len(tokens)
    counts = Counter(token for token in tokens)

    return counts, total

def open_file(path):
    '''
    Opens the file on path and counts the tokens, returning the url and token count.
    '''
    with zf.open(path, 'r') as f:
        data = json.load(f)
    url = data['url']
    content = data['content']

    soup = BeautifulSoup(content, 'html.parser')
    text = soup.get_text(strip=True)
    count, total = tokenize(text)
    return url, count, total

number_indexed= 0

r_index = defaultdict(list)
count = 0

with zipfile.ZipFile(PATH, "r") as zf:
    for filename in zf.namelist():
        if filename.endswith(".json"):
            url, word_count, _ = open_file(filename)
            print(url, len(word_count)) # debug

            for key in word_count.keys():
                r_index[key].append((filename, word_count[key])) #name for now, can also easily change it to be the file number. format: [file_name, count_of_word]

            number_indexed += 1

with open("index.json", "w") as f:
    json.dump(r_index, f)

total_size = os.path.getsize("index.json") / 1024 #get in bytes, then convert to KB
end_time = time.perf_counter()

delta_t = end_time - start_time
delta_min = (end_time - start_time) // 60
delta_sec = (end_time - start_time) % 60

print(f"Execution time: {end_time - start_time:.4f} seconds\n")
print("Number of indexed documents:", number_indexed)
print("Number of unique tokens:", len(r_index))
print("Size in KB:", total_size)

def update_stats():
    '''
    Writes to output.txt the analytics.
    '''
    temp = "output.txt.tmp"
    fin = "output.txt"

    with open(temp, "w") as outFile:
        outFile.write(f"Execution time: {delta_min} minutes, {delta_sec:.2f} seconds\n\n")
        outFile.write(f"Number of indexed documents: {number_indexed}\n")  # Using f-strings (Python 3.6+)
        outFile.write(f"Number of unique tokens: {len(r_index)}\n")
        outFile.write(f"Size in KB: {total_size:.2f}\n\n")

    os.replace(temp, fin)
update_stats()