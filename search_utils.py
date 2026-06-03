import re
import posixpath
from collections import Counter
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import hashlib, time, json
HASH_BITS = 64
hash_cache = set()

STOP_WORDS = {
    'a', 'the', 'and', 'of', 'to', 'in', 'is', 'that', 'for', 'on', 'with', 
    'it', 'as', 'by', 'at', 'an', 'from', 'this', 'about', 'which',
    'how', 'where', 'can', 'i', 'find', 'who', 'why'
}

def tokenize(text):
    '''
    Reads in text file and returns a normalized list.
    a token is a sequence of alphanumeric characters, independent of capitalization (so Apple, apple, aPpLe are the same token).
    returns Generator<Token>
    '''
    if not text:
        return [] # Safe exit if string is empty
    
    # Updated Regex: Captures alphanumeric words AND optional trailing '++' or '#'
    token_pattern = re.compile(r"[a-z0-9]+(?:\+\+|#)?")

    # 1. Extract all raw matched strings from the text in lowercase
    raw_tokens = [match.group() for match in token_pattern.finditer(text.lower())]
    
    # 2. Filter out stop words using an O(1) set check, then return the static list
    return [token for token in raw_tokens if token not in STOP_WORDS]

def hashify(token):
    '''
    Hash a token to a hash value. We use this instead of Python's built-in hash function because this is determinstic.
    '''
    md5_int = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
    return md5_int & ((1 << HASH_BITS) - 1)

def sim_hash(word_count: Counter):
    '''
    Given the word count, returns the SimHash of the page.
    '''
    bit_vector = [0] * HASH_BITS
    for token, weight in word_count.items():
        mask = 1
        hash_token = hashify(token)
        for i in range(HASH_BITS):
            if hash_token & mask:
                bit_vector[i] += weight
            else:
                bit_vector[i] -= weight
            mask <<= 1

    sim = 0
    mask = 1
    for i in range(HASH_BITS):
        if bit_vector[i] > 0:
            sim |= mask
        mask <<= 1

    return sim


def sim_hash_compare(sim_hash1, sim_hash2, threshold):
    '''
    Compares the 2 simhash and returns True if it meets the threshold and is similar, else False.
    '''
    x = sim_hash1 ^ sim_hash2
    diff_bits = x.bit_count()
    return (1 - diff_bits / HASH_BITS) > threshold


def is_similar(word_count: Counter, threshold=.95):
    '''
    Determines if a page is similar to previous pages based on word_count.
    '''
    sim = sim_hash(word_count)
    for other_hash in hash_cache:
        if sim_hash_compare(sim, other_hash, threshold):
            return True
    hash_cache.add(sim)
    return False

def canonicalize_url(url):
    """
    Cleans up URLs to prevent indexing duplicate pages with different parameters.
    """
    if not url:
        return ""
        
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]
        
    path = posixpath.normpath(parsed.path)
    if parsed.path.endswith('/') and not path.endswith('/'):
        path += '/'
        
    ignored_params = {'version', 'action', 'format', 'utm_source', 'sessionid', 'rev', 'limit'}
    query_pairs = parse_qsl(parsed.query)
    filtered_query = [(k, v) for k, v in query_pairs if k.lower() not in ignored_params]
    filtered_query.sort() # Ensure order variations result in identical keys
    query = urlencode(filtered_query)
    
    return urlunparse((scheme, netloc, path, parsed.params, query, ''))

def dump_json(data, dest, partial=False):
    """
    Dumps the dictionary into the destination file.
    """
    print(f"Dumping {dest}...")
    dumping_pre_t = time.perf_counter()
    with open(dest, "w") as f:
        if partial:
            for token in sorted(data.keys()):
                f.write(json.dumps({token: data[token]}) + "\n")
        else:
            json.dump(data, f)
    print(f"Dumping finished. {time.perf_counter() - dumping_pre_t:.4f} seconds.")
