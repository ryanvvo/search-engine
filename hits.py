import zipfile
import json
import re
import math
from collections import defaultdict
from urllib.parse import urljoin, urldefrag
from search_utils import dump_json

PATH = "developer.zip"

def _extract_links(base_url, html):
    """Return absolute URLs found in tags without fragments."""
    links = []
    for m in re.compile(r'<a[^>]+href=["\']([^"\'#][^"\']*)["\']',re.IGNORECASE).finditer(html):
        raw = m.group(1).strip()
        try:
            absolute = urljoin(base_url, raw)
            clean, _ = urldefrag(absolute)
            if clean.startswith(("http://", "https://")):
                links.append(clean)
        except:
            pass
    return links

class AdjIndex:
    """
    Builds an index from the zipfile
    """

    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        self.url_to_member = {}   # url : filename inside zip
        self.out_links = defaultdict(list)  # url : [url]
        self.in_links = defaultdict(list)  # url : [url]
        self._build_index()

    def _build_index(self):
        print(f"Building index from {self.zip_path}...")
        raw_links = {}   # url : raw outgoing urls

        with zipfile.ZipFile(self.zip_path, "r") as zf:
            members = [m for m in zf.namelist() if m.endswith(".json")]
            total = len(members)
            for i, member in enumerate(members):
                if i % 1000 == 0:
                    print(f"{i}/{total} indexing...")
                try:
                    with zf.open(member) as f:
                        data = json.load(f)
                    url = data.get("url", "").strip()
                    html = data.get("content", "")
                    if not url: continue

                    self.url_to_member[url] = member
                    raw_links[url] = _extract_links(url, html)
                except Exception: # skip bad files
                    pass

        # filter adjacency to corpus urls
        corpus = set(self.url_to_member.keys())
        for src, targets in raw_links.items():
            for tgt in targets:
                if tgt in corpus and tgt != src:
                    self.out_links[src].append(tgt)
                    self.in_links[tgt].append(src)

        print(f"index ready: {len(corpus)} pages, "
              f"{sum(len(v) for v in self.out_links.values())} edges")

    def all_urls(self):
        return set(self.url_to_member.keys())

def hit_alg(nodes, index, max_iter=50, tol=1e-6):
    nodes = list(nodes)
    n = len(nodes)
    if n == 0:
        return {}

    print(f"running iterations on {n} nodes …")

    idx = {url: i for i, url in enumerate(nodes)} # {url : i}
    node_set = set(nodes)

    # initialise scores
    hub  = [1.0] * n
    auth = [1.0] * n

    # pre-build adjacency restricted to the subgraph
    out = [[] for _ in range(n)]
    inn = [[] for _ in range(n)]
    for url in nodes:
        i = idx[url]
        for tgt in index.out_links.get(url, []):
            if tgt in node_set:
                j = idx[tgt]
                out[i].append(j)
                inn[j].append(i)

    for iteration in range(max_iter):
        # authority update
        new_auth = [0.0] * n
        for i in range(n):
            for j in inn[i]:
                new_auth[i] += hub[j]

        # hub update
        new_hub = [0.0] * n
        for i in range(n):
            for j in out[i]:
                new_hub[i] += new_auth[j]

        # l2 normalize
        auth_norm = math.sqrt(sum(x * x for x in new_auth)) or 1.0
        hub_norm  = math.sqrt(sum(x * x for x in new_hub))  or 1.0
        new_auth = [x / auth_norm for x in new_auth]
        new_hub  = [x / hub_norm  for x in new_hub]

        # convergence check
        delta = math.sqrt(sum((new_auth[i] - auth[i]) ** 2 + (new_hub[i] - hub[i]) ** 2 for i in range(n)))
        auth, hub = new_auth, new_hub
        print(f"iter {iteration + 1:3d}  d={delta:.2e}")
        if delta < tol:
            print("converged")
            break
    else:
        print("did not converge")

    return {nodes[i]: {"hub": hub[i], "authority": auth[i]} for i in range(n)}

def pagerank(nodes, index, damping=0.85, max_iter=50, tol=1e-6):
    nodes = list(nodes)
    n = len(nodes)

    if n == 0:
        return {}

    print(f"running PageRank on {n} nodes...")

    idx = {url: i for i, url in enumerate(nodes)}
    node_set = set(nodes)

    # build adjacency
    out = [[] for _ in range(n)]
    for url in nodes:
        i = idx[url]

        for tgt in index.out_links.get(url, []):
            if tgt in node_set:
                out[i].append(idx[tgt])

    # initial ranks
    rank = [1.0 / n] * n
    base = (1.0 - damping) / n
    for iteration in range(max_iter):
        new_rank = [base] * n
        dangling_mass = 0.0

        # distribute rank
        for i in range(n):
            if not out[i]:
                dangling_mass += rank[i]
                continue
            share = damping * rank[i] / len(out[i])
            for j in out[i]:
                new_rank[j] += share

        # distribute dangling pages
        dangling_share = damping * dangling_mass / n
        for i in range(n):
            new_rank[i] += dangling_share

        # convergence
        delta = math.sqrt(sum((new_rank[i] - rank[i]) ** 2 for i in range(n)))
        rank = new_rank
        print(f"iter {iteration + 1:3d}  d={delta:.2e}")

        if delta < tol:
            print("converged")
            break

    return {nodes[i]: rank[i] for i in range(n)}

def authority_scores(results):
    return {url: scores["authority"] for url, scores in results.items()}

def hub_scores(results):
    return {url: scores["hub"] for url, scores in results.items()}

if __name__ == "__main__":
    index = AdjIndex(PATH)
    nodes = index.all_urls()
    hit_results = hit_alg(nodes, index)
    pr_results = pagerank(nodes, index)

    dump_json(hit_results, "hits_results.json")
    dump_json(pr_results, "pr_results.json")