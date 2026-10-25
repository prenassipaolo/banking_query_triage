import string


def _tokenize(text):

    # Stopwords list can be expanded as needed
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "for",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
    }

    # Replace punctuation with spaces
    text = text.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    # Split the text into words
    tokens = text.split()
    # Keep only alphanumeric characters
    tokens = [token for token in tokens if token.isalnum()]
    # Convert to lowercase
    tokens = [token.lower() for token in tokens]
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords]
    return tokens


def _get_vocabulary(corpus):
    all_words = set()
    for tokens in corpus:
        all_words.update(tokens)
    return sorted(all_words)


def _get_word_counts(list_of_tokens, vocabulary):
    word_counts = {}
    for token in vocabulary:
        word_counts[token] = 0
    for tokens in list_of_tokens:
        for token in tokens:
            if token in word_counts:
                word_counts[token] += 1
    return word_counts
