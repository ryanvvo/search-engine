import zipfile
import json
import re
import os
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)



number_indexed= 0

r_index = defaultdict(list)
count = 0
with zipfile.ZipFile("analyst.zip", "r") as zf:        #currently just looking at analyst
    for filename in zf.namelist():
        if filename.endswith(".json"):
            with zf.open(filename) as f:
                data = json.load(f)

            html = data['content']
            soup = BeautifulSoup(html, "html.parser")
            
            text = soup.get_text(separator=" ", strip=True)
            words = re.findall(r"\b\w+\b", text.lower())
            word_count = Counter(words)
            #print(word_count)

            for key in word_count.keys():
                r_index[key].append([filename, word_count[key]]) #name for now, can also easily change it to be the file number. format: [file_name, count_of_word]

            number_indexed += 1

with open("index.json", "w") as f:
    json.dump(r_index, f)

total_size = os.path.getsize("index.json") / 1024 #get in bytes, then convert to KB


print("Number of indexed documents:", number_indexed)
print("Number of unique tokens:", len(r_index))
print("Size in KB:", total_size)

