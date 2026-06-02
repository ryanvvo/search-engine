from flask import Flask, request, jsonify, render_template
import json
from search_query import query_search, OFFSET_PATH, MAP_PATH
import time
import webbrowser
from pathlib import Path
from collections import defaultdict
app = Flask(__name__)

# Homepage route
with open(MAP_PATH, "r") as inFile:
    mapping = json.load(inFile)
with open(OFFSET_PATH, "r") as inFile:
    offsets = json.load(inFile)
with open("two_gram_" + MAP_PATH, "r") as inFile:
    two_gram_mapping = json.load(inFile)
with open("two_gram_" + OFFSET_PATH, "r") as inFile:
    two_gram_offsets = json.load(inFile)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "")

    start = time.perf_counter()
    results = query_search(query, mapping, offsets)
    if Path("two_gram_index.json").exists():
        two_gram_results = query_search(query, two_gram_mapping, two_gram_offsets, two_gram=True)
        score_map = defaultdict(float)
        for score, url in results + two_gram_results:
            score_map[url] += score
        results = [(score, url) for url, score in score_map.items()]

    results.sort(reverse=True, key=lambda x: x[0])

    return jsonify({
        "results": results,
        "time": f"{time.perf_counter() - start:.4f}"
    })

if __name__ == "__main__":
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=True, use_reloader=False)