import json
import itertools
import random
import choix
from typing import List
import pandas as pd
import numpy as np
from loguru import logger
from collections import defaultdict


# This function is copied from the main repo of robin
def uniformly_random_pairs(
    n_hypotheses: int, seed: int = 621, n_games: int | None = None
) -> List[List[int]]:
    random.seed(seed)
    min_hypotheses = 2

    if n_hypotheses < min_hypotheses:
        if n_games is not None and n_games > 0:
            raise ValueError(
                f"Cannot generate {n_games} pairs with distinct elements when"
                f" n_hypotheses={n_hypotheses} < 2."
            )
        return []

    max_possible_games = n_hypotheses * (n_hypotheses - 1) // 2
    if n_games is None:
        n_games = min(300, max_possible_games)
    elif n_games < 1:
        raise ValueError("n_games cannot be less than 1.")

    if n_games > max_possible_games:
        logger.warning(
            f"n_games ({n_games}) exceeds max possible ({max_possible_games}). "
            f"Setting n_games to {max_possible_games}."
        )
        n_games = max_possible_games

    all_possible_unordered_pairs = list(itertools.combinations(range(n_hypotheses), 2))
    sampled_pairs = random.sample(all_possible_unordered_pairs, n_games)

    return [list(pair) for pair in sampled_pairs]


def crew_output_to_dataframe(crew_output) -> pd.DataFrame:
    # Parse the raw JSON string
    parsed_json = json.loads(crew_output.raw)

    # Extract relevant fields from each item in the list
    records = []
    for entry in parsed_json:
        records.append({
            "analysis": entry["Analysis"],
            "reasoning": entry["Reasoning"],
            "winner_name": entry["Winner"]["name"],
            "winner_id": entry["Winner"]["id"],
            "loser_name": entry["Loser"]["name"],
            "loser_id": entry["Loser"]["id"]
        })

    # Convert to DataFrame
    return pd.DataFrame(records)


def crew_output_to_json(crew_output) -> List[dict]:
    """
    Converts CrewAI output to a list of dictionaries.
    """

    parsed_json = json.loads(crew_output.raw)
    records = []

    for entry in parsed_json:
        records.append({
            "analysis": entry["Analysis"],
            "reasoning": entry["Reasoning"],
            "winner_name": entry["Winner"]["name"],
            "winner_id": entry["Winner"]["id"],
            "loser_name": entry["Loser"]["name"],
            "loser_id": entry["Loser"]["id"]
        })

    return records


def load_hypotheses_json_to_df(json_path: str) -> pd.DataFrame:
    """
    Load hypotheses from JSON into a flat DataFrame with 'id' and 'assay_name' columns.
    """
    try:
        with open(json_path, "r") as f:
            raw = json.load(f)

        flattened = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(f"Unexpected format at index {i}: {item}")

            assay_name, data = list(item.items())[0]
            row = {"id": i, "assay_name": assay_name}
            row.update(data)
            flattened.append(row)

        return pd.DataFrame(flattened)

    except Exception as e:
        logger.error(f"Failed to load JSON file {json_path}: {e}")
        raise


def compute_btl_scores(df: pd.DataFrame, item_col: str = "winner_id", opponent_col: str = "loser_id") -> pd.DataFrame:
    """
    Compute BTL scores from a dataframe of pairwise comparisons.

    Parameters:
    - df: pandas DataFrame with at least 'winner_id' and 'loser_id' columns.
    - item_col: Column name for winner IDs (default = 'winner_id').
    - opponent_col: Column name for loser IDs (default = 'loser_id').

    Returns:
    - DataFrame with columns: ,'item_id' and 'btl_score' (normalized).
    """

    # Step 1: Extract pairwise outcomes
    comparisons = list(zip(df[item_col], df[opponent_col]))

    # Step 2: Get number of unique items
    unique_items = sorted(set(df[item_col]).union(set(df[opponent_col])))
    n_items = max(unique_items) + 1

    # Step 3: Fit BTL model choix implementation - config used in Robin original repo
    theta = choix.lsr_pairwise(n_items=n_items, data=comparisons, alpha=0.1)

    # Step 4: Normalize to get scores between 0 and 1
    theta_exp = np.exp(theta)
    btl_scores = theta_exp / theta_exp.sum()

    return pd.DataFrame({
        "item_id": list(range(n_items)),
        "btl_score": btl_scores
    }).sort_values(by="btl_score", ascending=False).reset_index(drop=True)


def compute_elo_scores(df: pd.DataFrame, item_col="winner_id", opponent_col="loser_id", K=32, base_rating=1000) -> pd.DataFrame:
    """
    Computes Elo ratings for items from pairwise comparison data.

    Parameters:
    - df: DataFrame with at least 'winner_id' and 'loser_id' columns.
    - K: learning rate (default 32)
    - base_rating: starting Elo rating for all items

    Returns:
    - DataFrame with columns: 'item_id', 'elo_rating'
    """
    ratings = defaultdict(lambda: base_rating)

    for _, row in df.iterrows():
        winner = row[item_col]
        loser = row[opponent_col]

        R_w = ratings[winner]
        R_l = ratings[loser]

        # Expected scores
        E_w = 1 / (1 + 10 ** ((R_l - R_w) / 400))
        E_l = 1 - E_w

        # Update ratings
        ratings[winner] += K * (1 - E_w)
        ratings[loser]  += K * (0 - E_l)

    return pd.DataFrame([
        {"item_id": item, "elo_rating": rating}
        for item, rating in sorted(ratings.items(), key=lambda x: -x[1])
    ])
