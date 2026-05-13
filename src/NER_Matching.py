import pandas as pd
import re
from rapidfuzz import fuzz
from rapidfuzz import process
import concurrent.futures
from datetime import datetime
from collections import defaultdict


def combine_adjacent_blocks(text, block_pattern, print_pattern):
    result = re.sub(rf"({block_pattern} {block_pattern})", print_pattern, text)
    return result


def split_names(names_list):
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

    return split_name_list


def split_review(review):
    """Clean reviews a bit and turn them into word lists."""

    ## general clean
    review_cleaned = re.sub(r"[’']s\b", "", review)
    review_cleaned = re.sub(r"[^a-zA-Z0-9?!.;, ]", "", review_cleaned)

    # Replace slashes and periods with spaces to treat them as word delimiters, then split
    review_cleaned = re.sub(r"[/.]", " ", review_cleaned)
    words = review_cleaned.split()

    return words


def extract_words_from_reviews(review_list):
    """
    Get set of unique words from all the reviews for a movie
    """
    all_words = set()
    for review in review_list:
        words = split_review(review)  # Split the review into words
        all_words.update(map(lambda w: w.lower(), words))
    return all_words


def add_fuzzy_matches(names, vocabulary, placeholder, threshold=90, verbose=None):
    """
    Generate matches for all the words in a vocab against all the named entitites
    """
    fuzzy_match_map = defaultdict(list)
    for name in names:
        matches = process.extract(
            name, vocabulary, limit=50
        )  # Get the top 30 fuzzy matches for each name
        if verbose == 2:
            print(name, " : ", matches)
        for match, score in matches:
            if score >= threshold:
                fuzzy_match_map[match.lower()].append(placeholder)
            else:
                break  # Stop matching once a score below the threshold is found
    if verbose:
        print(fuzzy_match_map)
    return fuzzy_match_map


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
        sentence_chunk = " ".join(
            words[i : i + len(title.split())]
        )  # Grab chunk based on title length

        # If the chunk is similar enough to the title, replace it
        score = fuzz.ratio(title.lower(), sentence_chunk.lower())
        if score >= threshold:
            modified_review.append("[TITLE]")
            i += len(sentence_chunk.split())  # Skip the matched chunk length
        else:
            modified_review.append(word)
            i += 1

    return " ".join(modified_review)


# Main function to replace actors, directors, roles in reviews with placeholders
def replace_actors_with_placeholder(
    review_list, actor_list, director_list, title, role_list, verbose=None
):
    """
    main func to replace actors, directors, and roles with placeholders
    """
    review_list = ast.literal_eval(review_list)  ## in case it reads in as string

    # Split the names into a list of individual names
    actor_list = split_names(actor_list)
    director_list = split_names(director_list)
    role_list = filter_names_from_dict(split_names(role_list))

    # Extract vocabulary
    vocabulary = extract_words_from_reviews(review_list)

    # Generate fuzzy matches for actors, directors, and roles
    fuzzy_match_map = defaultdict(list)
    fuzzy_match_map.update(
        add_fuzzy_matches(
            actor_list, vocabulary, "[ACTOR]", threshold=95, verbose=verbose
        )
    )
    fuzzy_match_map.update(
        add_fuzzy_matches(
            director_list, vocabulary, "[DIRECTOR]", threshold=95, verbose=verbose
        )
    )
    fuzzy_match_map.update(
        add_fuzzy_matches(
            role_list, vocabulary, "[ROLE]", threshold=95, verbose=verbose
        )
    )
    fuzzy_match_map.update(
        add_fuzzy_matches(
            role_list, vocabulary, "[ROLE]", threshold=95, verbose=verbose
        )
    )
    fuzzy_match_map.update(
        add_fuzzy_matches(
            role_list, vocabulary, "[ROLE]", threshold=95, verbose=verbose
        )
    )

    updated_reviews = []
    for review in review_list:
        updated_words = []
        for word in split_review(review):
            normalized_word = word.lower()

            # Check if the word matches any fuzzy match in the map
            if normalized_word in fuzzy_match_map:
                updated_words.append(
                    fuzzy_match_map[normalized_word][0]
                )  # just use first match
            else:
                updated_words.append(word)

        # Join the updated words back into a single string
        updated_review = " ".join(updated_words)
        updated_review = replace_title_in_review(title=title, review=updated_review)

        # Combine adjacent blocks of placeholders
        updated_review = combine_adjacent_blocks(
            updated_review, r"\[ACTOR\]", "[ACTOR]"
        )
        updated_review = combine_adjacent_blocks(updated_review, r"\[ROLE\]", "[ROLE]")
        updated_review = combine_adjacent_blocks(
            updated_review, r"\[DIRECTOR\]", "[DIRECTOR]"
        )

        updated_reviews.append(updated_review)
        if verbose:
            print(f"{review}\n{updated_review}'\n'{'--' * 50}")

    return updated_reviews


def get_NER_all(data, chunksize=100, begin=0, length=None, output_file=None):
    df = data.copy()

    if not output_file:
        output_file = f"{root}/Data/2020_trope_data/Scraped_Data/NER_cleaned.csv"

    if not length:
        length = len(df)

    print(
        f"Scraping studios for movies from {begin} to {length} with chunksize={chunksize}"
    )

    studios = []

    # Parallelizing with ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        for start in range(begin, length, chunksize):
            current_time = datetime.now()
            print(
                f"started {start} to {start + chunksize}  at  "
                + current_time.strftime("%Y-%m-%d %H:%M:%S")
            )
            chunk = df.iloc[start : start + chunksize].copy()

            chunk["NER_cleaned_data"] = [[] for _ in range(len(chunk))]

            # Submit tasks for fetching comments and budget separately
            futures = {
                index: executor.submit(replace_actors_with_placeholder, reviews)
                for index, reviews in chunk["url"].items()
            }

            for index, future in futures.items():
                studio = future.result()

                # Directly assign the actors and roles to the correct index in the chunk
                chunk.at[index, "studio"] = studio  ## note we use .at instead of .loc

            chunk.to_csv(
                output_file,
                mode="a",
                header=not pd.io.common.file_exists(output_file),
                index=False,
            )
            studios.clear()

            current_time = datetime.now()
            print(
                f"finished {start} to {start + chunksize}  at  "
                + current_time.strftime("%Y-%m-%d %H:%M:%S")
            )

    print(f"Scraped all studios from {begin} to {length} with chunksize={chunksize}")
