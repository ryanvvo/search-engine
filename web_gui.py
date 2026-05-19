from flask import Flask, request, jsonify, render_template
import json
from search_query import query_search, OFFSET_PATH, MAP_PATH
import time
import webbrowser
app = Flask(__name__)

# Homepage route
with open(MAP_PATH, "r") as inFile:
    mapping = json.load(inFile)
with open(OFFSET_PATH, "r") as inFile:
    offsets = json.load(inFile)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "")

    start = time.perf_counter()
    results = query_search(query, mapping, offsets)

    return jsonify({
        "results": results[:5],
        "time": f"{time.perf_counter() - start:.4f}"
    })

if __name__ == "__main__":
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=True, use_reloader=False)