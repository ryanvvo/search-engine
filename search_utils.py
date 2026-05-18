import re

def tokenize(text):
    '''
    Reads in text file and returns a normalized list.
    a token is a sequence of alphanumeric characters, independent of capitalization (so Apple, apple, aPpLe are the same token).
    returns Generator<Token>
    '''

    return (match.group() for match in re.finditer(r"[a-z0-9]+", text.lower()))