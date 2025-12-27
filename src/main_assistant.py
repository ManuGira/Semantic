import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join as pjoin
from common import load_model, load_most_frequent_words, compute_heatmap_matrix


def plot_matrix(correlation_matrix, words):
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, xticklabels=words, yticklabels=words, annot=True, cmap="coolwarm")
    plt.title("Word Distance Matrix")
    plt.show()


def compute_word_probabilities(heatmap_matrix: np.ndarray, words: list[str],
                               observed_words: list[str], observed_scores: list[int]):
    """
    Compute probability for each word to be the secret word based on observed rankings.

    Args:
        heatmap_matrix: NxN similarity matrix between all words
        words: list of all N words
        observed_words: words already tested
        observed_scores: their corresponding scores -100 to 100, higher is better

    Returns:
        Array of probabilities for each word
    """

    observed_indexes = [words.index(word) for word in observed_words]
    not_observed_indexes = [i for i in range(len(words)) if i not in observed_indexes]

    # Use np.ix_ to properly index rows and columns
    sub_matrix = heatmap_matrix[np.ix_(not_observed_indexes, observed_indexes)]

    # remove rows of observed words
    words = [word for word in words if word not in observed_words]

    # normalizes each lines of the matrix so that it sums to 1
    sub_matrix = sub_matrix / np.sum(sub_matrix ** 2, axis=1, keepdims=True)

    # Normalize observed scores to sum to 1
    observed_scores = np.array(observed_scores)
    observed_scores = observed_scores / np.sum(observed_scores)

    suggestions_score = np.dot(sub_matrix, observed_scores)

    # sort scores to get a ranked list of words
    ranked_indexes = np.argsort(-suggestions_score)
    ranked_scores = suggestions_score[ranked_indexes]
    ranked_words = [words[i] for i in ranked_indexes]

    return ranked_words, ranked_scores


def play_game_assistant(heatmap_matrix, words):
    observed_words = []
    observed_scores = []
    while True:
        word = input("Enter observed word: ").strip()
        if word.lower() == '_':
            break
        if word not in words:
            print(f"Word '{word}' not in the list. Please try again.")
            continue

        while True:
            try:
                score = float(input(f"Enter score for '{word}': "))
                break
            except ValueError:
                print("Invalid input. Please enter a numeric score.")

        observed_words.append(word)
        observed_scores.append(score)

        if len(observed_words) < 2:
            continue

        ranked_words, ranked_scores = compute_word_probabilities(heatmap_matrix, words, observed_words, observed_scores)
        print(f"\nAfter observing '{word}' with score {score}:")
        for word, score in list(zip(ranked_words, ranked_scores))[:10]:  # show top 10 suggestions
            print(f"{word}: {score:.4f}")


def main():
    N = 3100
    print(f"Using first {N} words.")
    words = load_most_frequent_words(N)

    # compute heatmap matrix if not already saved
    here = os.path.abspath(os.path.dirname(__file__))
    cached_matrix_file = pjoin(here, "..", ".cache", f"heatmap_matrix_{N}.npy")
    if os.path.exists(cached_matrix_file):
        print("Loading precomputed heatmap matrix...")
        tick = time.time()
        heatmap_matrix = np.load(cached_matrix_file)
        tock = time.time()
        print(f"Correlation matrix loaded in {tock - tick:.02f} seconds.")
    else:
        print("Loading model...")
        tick = time.time()
        model = load_model()
        tock = time.time()
        print(f"Model loaded in {tock - tick:.02f} seconds.")

        print("Computing heatmap matrix...")
        tick = time.time()
        heatmap_matrix = compute_heatmap_matrix(model, words)
        tock = time.time()
        print(f"Heatmap matrix computed in {tock - tick:.02f} seconds.")
        # save heatmap matrix for future use
        os.makedirs(os.path.dirname(cached_matrix_file), exist_ok=True)
        np.save(cached_matrix_file, heatmap_matrix)

    if N <= 50:
        print("Plotting correlation matrix...")
        plot_matrix(heatmap_matrix, words)

    print("Starting game...")
    play_game_assistant(heatmap_matrix, words)


if __name__ == "__main__":
    main()

