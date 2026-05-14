import pandas as pd
import re
from rapidfuzz import process
from collections import defaultdict
from rapidfuzz.distance import DamerauLevenshtein as DL


def combine_adjacent_blocks(text, block_pattern, print_pattern):
    result = re.sub(rf"({block_pattern} {block_pattern})", print_pattern, text)
    return result


def split_names(names_list) -> set[str]:
    """Split eg actor or director names into two ([TOM HANKS ]=> [Tom, Hanks])"""
    split_name_list = []
    for name in names_list:
        parts = name.split()
        if len(parts) > 1:
            # Only take the first and last names
            first_name = parts[0].lower()
            last_name = parts[-1].lower()
            split_name_list.append(first_name)
            split_name_list.append(last_name)
        else:
            # If only one name is provided (e.g., a single first name), just add it
            split_name_list.append(parts[0].lower())

    return set(split_name_list)


def split_review(review: str):
    """Clean reviews a bit and turn them into word lists."""

    if review is None:
        return []
    ## general clean
    review_cleaned = re.sub(r"[’']s\b", "", review)
    review_cleaned = re.sub(r"[^a-zA-Z0-9?!.;, ]", "", review_cleaned)

    # Replace slashes and periods with spaces to treat them as word delimiters, then split
    review_cleaned = re.sub(r"[!/.:]", " ", review_cleaned)
    words = review_cleaned.split()

    return words


def get_vocab(review_list: pd.Series):
    """
    Get set of unique words from all the reviews for a movie
    """
    all_words = set()
    for review in review_list:
        words = split_review(review)  # Split the review into words
        all_words.update(map(lambda w: w.lower(), words))
    return all_words


def replace_title_in_review(title, review, threshold=80):
    """
    Func to replace the title in a review. For titles we already have the 'vocab' and can match quickly.
    """
    words = review.split()

    # Check each word to see if it fuzzy matches the title, and replace it if it does
    modified_review = []
    i = 0
    while i < len(words):  ## while loop cuz iteration length shifts
        word = words[i]

        # Check fuzzy match of the word with the title, allow for punctuation in between
        sentence_chunk = " ".join(words[i : i + len(title.split())])
        score = DL.normalized_similarity(title.lower(), sentence_chunk.lower())
        if score >= threshold:
            modified_review.append("[TITLE]")
            i += len(sentence_chunk.split())  # Skip the matched chunk length
        else:
            modified_review.append(word)
            i += 1

    return " ".join(modified_review)


def add_fuzzy_matches(names, vocab, placeholder) -> defaultdict:
    fuzzy_match_map = defaultdict(list)
    for name in names:
        threshold = max(0.85, 1.0 - 0.02 * len(name))
        matches = process.extract(
            name, vocab, scorer=DL.normalized_similarity, score_cutoff=threshold
        )
        for match, _, _ in matches:
            fuzzy_match_map[match.lower()].append(placeholder)
    return fuzzy_match_map


def get_general_fuzzy_map(review_df, meta_df) -> defaultdict:
    vocab = get_vocab(review_df["review"])

    fuzzy_match_map = defaultdict(list)
    for names, label in [
        (split_names(meta_df["actors"].explode().dropna()), "[ACTOR]"),
        (split_names(meta_df["directors"].explode().dropna()), "[DIRECTOR]"),
        (meta_df["studio"].explode().dropna(), "[STUDIO]"),
        (list(meta_df["year"][0]), "[YEAR]"),
    ]:
        fuzzy_match_map.update(add_fuzzy_matches(names, vocab, label))
    return fuzzy_match_map


def replace_NER(df: pd.DataFrame, fuzzy_match_map: defaultdict[str, list]) -> pd.Series:
    title = df["Title"][0]
    updated_reviews = []
    for review in df["review"]:
        updated_words = [
            fuzzy_match_map.get(word.lower(), [word])[0]
            for word in split_review(review)
        ]
        updated_review = " ".join(updated_words)
        updated_review = replace_title_in_review(title, updated_review)

        for rblk, blk in [
            (r"\[ACTOR\]", "[ACTOR]"),
            (r"\[ROLE\]", "[ROLE]"),
            (r"\[DIRECTOR\]", "[DIRECTOR]"),
            (r"\[STUDIO\]", "[STUDIO]"),
            (r"\[TITLE\]", "[TITLE]"),
        ]:
            updated_review = combine_adjacent_blocks(updated_review, rblk, blk)
        updated_reviews.append(updated_review)
    return pd.Series(updated_reviews)


def NER_pipeline(df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    fuzzy_match_map = get_general_fuzzy_map(df, meta_df)
    return replace_NER(df, fuzzy_match_map)
