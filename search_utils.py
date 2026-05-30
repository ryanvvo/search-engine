import re
from collections import Counter
import hashlib
HASH_BITS = 64
hash_cache = set()

def tokenize(text):
    '''
    Reads in text file and returns a normalized list.
    a token is a sequence of alphanumeric characters, independent of capitalization (so Apple, apple, aPpLe are the same token).
    returns Generator<Token>
    '''

    return (match.group() for match in re.finditer(r"[a-z0-9]+", text.lower()))

def hashify(token):
    '''
    Hash a token to a hash value. We use this instead of Python's built-in hash function because this is determinstic.
    '''
    return int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)

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