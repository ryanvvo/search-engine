from flask import Flask, request, jsonify, render_template
import json
from search_query import query_search, OFFSET_PATH, MAP_PATH, evaluate_position, filter_and_diversify_results
import time
import webbrowser
from pathlib import Path
from collections import defaultdict
app = Flask(__name__)

HAS_TWO_GRAM = Path("two_gram_index.json").exists()
HAS_POSITIONAL = Path("positional_index.json").exists()
HAS_HITS = Path("hits_results.json").exists()
HAS_PR = Path("pr_results.json").exists()

# Homepage route
with open(MAP_PATH, "r") as inFile:
    print("Loading mapping...")
    mapping = json.load(inFile)
with open(OFFSET_PATH, "r") as inFile:
    print("Loading offsets...")
    offsets = json.load(inFile)

if HAS_TWO_GRAM:
    print("Found two-gram index.")
    with open("two_gram_" + MAP_PATH, "r") as inFile:
        print("Loading two-gram mapping...")
        two_gram_mapping = json.load(inFile)
    with open("two_gram_" + OFFSET_PATH, "r") as inFile:
        print("Loading two-gram offsets...")
        two_gram_offsets = json.load(inFile)

if HAS_POSITIONAL:
    print("Positional index found.")
    with open("positional_index_inv_" + MAP_PATH, "r") as inFile:
        print("Loading positional-index inverse mapping...")
        positional_inverse_index_mapping = json.load(inFile)
    with open("positional_index_" + OFFSET_PATH, "r") as inFile:
        print("Loading positional-index offsets...")
        positional_index_offsets = json.load(inFile)

if HAS_HITS:
    print("Hits results found...")
    with open("hits_results.json", 'r') as inFile:
        print("Loading hits-results...")
        hits_results = json.load(inFile)

if HAS_PR:
    print("PR results found...")
    with open("pr_results.json", 'r') as inFile:
        print("Loading pr-results...")
        pr_results = json.load(inFile)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "")

    start = time.perf_counter()
    results = query_search(query, mapping, offsets)

    if HAS_TWO_GRAM: # Add results of two-gram if it exists
        two_gram_results = query_search(query, two_gram_mapping, two_gram_offsets, two_gram=True)
        score_map = defaultdict(float)
        for score, url in results + two_gram_results:
            score_map[url] += score
        results = [(score, url) for url, score in score_map.items()]

    if HAS_POSITIONAL: # Add results of positional if it exists
        evaluate_position(query, results, positional_inverse_index_mapping, positional_index_offsets)

    if HAS_HITS:
        for i, (score, url) in enumerate(results):
            results[i] = (score + score * hits_results[url]['authority'], url)

    if HAS_PR:
        for i, (score, url) in enumerate(results):
            results[i] = (score + score * pr_results[url], url)

    results.sort(reverse=True, key=lambda x: x[0])
    results = filter_and_diversify_results(results, max_per_domain=3)

    return jsonify({
        "results": results,
        "time": f"{time.perf_counter() - start:.4f}"
    })

if __name__ == "__main__":
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=True, use_reloader=False)
