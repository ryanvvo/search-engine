from flask import Flask, request, jsonify, render_template
import json
from search_query import query_search, OFFSET_PATH, MAP_PATH, evaluate_position, filter_and_diversify_results
from search_utils import tick_timer
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
    start = time.perf_counter()
    mapping = json.load(inFile)
    tick_timer("Time to get mapping: ", start)
with open(OFFSET_PATH, "r") as inFile:
    start = time.perf_counter()
    print("Loading offsets...")
    offsets = json.load(inFile)
    tick_timer("Time to get offsets: ", start)

if HAS_TWO_GRAM:
    print("Found two-gram index.")
    with open("two_gram_" + MAP_PATH, "r") as inFile:
        start = time.perf_counter()
        print("Loading two-gram mapping...")
        two_gram_mapping = json.load(inFile)
        tick_timer("Time to get two-gram mapping: ", start)
    with open("two_gram_" + OFFSET_PATH, "r") as inFile:
        print("Loading two-gram offsets...")
        start = time.perf_counter()
        two_gram_offsets = json.load(inFile)
        tick_timer("Time to get two-gram offsets: ", start)

if HAS_POSITIONAL:
    print("Positional index found.")
    with open("positional_index_inv_" + MAP_PATH, "r") as inFile:
        print("Loading positional-index inverse mapping...")
        start = time.perf_counter()
        positional_inverse_index_mapping = json.load(inFile)
        tick_timer("Time to get positional-index inverse mapping: ", start)
    with open("positional_index_" + OFFSET_PATH, "r") as inFile:
        start = time.perf_counter()
        print("Loading positional-index offsets...")
        positional_index_offsets = json.load(inFile)
        tick_timer("Time to get positional-index offsets: ", start)

if HAS_HITS:
    start = time.perf_counter()
    print("Hits results found...")
    with open("hits_results.json", 'r') as inFile:
        print("Loading hits-results...")
        hits_results = json.load(inFile)
    tick_timer("Time to get hits results: ", start)

if HAS_PR:
    start = time.perf_counter()
    print("PR results found...")
    with open("pr_results.json", 'r') as inFile:
        print("Loading pr-results...")
        pr_results = json.load(inFile)
    tick_timer("Time to get pr results: ", start)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "")

    first_start = time.perf_counter()
    results = query_search(query, mapping, offsets)
    start = tick_timer("Time to get results: ", first_start)
    if HAS_TWO_GRAM: # Add results of two-gram if it exists
        two_gram_results = query_search(query, two_gram_mapping, two_gram_offsets, two_gram=True)
        score_map = defaultdict(float)
        for score, url in results + two_gram_results:
            score_map[url] += score
        results = [(score, url) for url, score in score_map.items()]
        start = tick_timer("Time to add two_gram: ", start)
    if HAS_POSITIONAL: # Add results of positional if it exists
        evaluate_position(query, results, positional_inverse_index_mapping, positional_index_offsets)
        start = tick_timer("Time to add positional: ", start)
    if HAS_HITS:
        for i, (score, url) in enumerate(results):
            results[i] = (score + score * hits_results[url]['authority'], url)
        start = tick_timer("Time to add hits: ", start)
    if HAS_PR:
        for i, (score, url) in enumerate(results):
            results[i] = (score + score * pr_results[url], url)
        start = tick_timer("Time to add pr: ", start)
    results.sort(reverse=True, key=lambda x: x[0])
    start = tick_timer("Time to sort: ", start)
    results = filter_and_diversify_results(results, max_per_domain=3)
    start = tick_timer("Time to add filter: ", start)
    return jsonify({
        "results": results,
        "time": f"{time.perf_counter() - first_start:.4f}"
    })

if __name__ == "__main__":
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=True, use_reloader=False)
