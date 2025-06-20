import json
from typing import  Optional
from crewai.project import CrewBase
import pandas as pd
from loguru import logger


from .file_system import FileSystem
from .configuration import RobinConfiguration
from .utils import (crew_output_to_dataframe,
                   crew_output_to_json ,
                   load_hypotheses_json_to_df,
                   uniformly_random_pairs,
                   compute_elo_scores,
                   compute_btl_scores)

# TODO: have a function/ class that can help with different ranking method
class HypothesesRanker:

    """ This class can do the following:
    - come up with the uniform_random_pairs
    - handle different ranking methods (implemented in them)
    - perform the ranking using pairwise comparison
    - store the output

    Methods:
    - run pairwise_comparison

    outputs:
    ranking_output: pd.DataFrame

    """

    def __init__(self,
                 robin_config:RobinConfiguration,
                 ranker_crew_instance: CrewBase,
                 ranking_system: str,
                 file_system: FileSystem,
                 hypotheses_dataframe:  Optional[pd.DataFrame] = None,
                 ):
        self.robin_config = robin_config
        self.ranker_crew_instance = ranker_crew_instance
        self.ranking_system = ranking_system
        self.file_system = file_system

        if hypotheses_dataframe is not None:
            self.hypotheses_df = hypotheses_dataframe
        else:
            hypotheses_path = self.file_system.get_hypotheses_output_path_json()
            self.hypotheses_df = load_hypotheses_json_to_df(hypotheses_path)

        self.json_list = self.hypotheses_df.to_dict(orient="records")
        self.ranker_result = None

    def compare_single_pair(self, idx1: int, idx2: int) -> dict:
        """Runs CrewAI for a single comparison pair and returns the result.
        Inputs:
        Pair of Assays,
        where each assay looks like {assay_name:ste
        Evaluation:str,
        Previous Use: str
        Biomedical Evidence: str,
        Overall Assessment: str}

        Output: JSON with winner,loser, analysis, reasoning"""
        hypotheses_subset = json.dumps([self.json_list[idx1], self.json_list[idx2]], indent=2)

        inputs_single_pair = {
            "topic": self.robin_config.disease_name,
            "num_assays": self.robin_config.num_assays,
            "hypotheses": hypotheses_subset,
            "pair_list": [[0, 1]]
        }

        # crew_instance = self.ranker_crew_instance(llm=self.robin_config.llm_client.get_llm())
        crew = self.ranker_crew_instance.crew()

        result = crew.kickoff(inputs=inputs_single_pair)
        return result

    def run_all_comparison(self) -> pd.DataFrame:
        """Iterates over all uniform pairs and stores the results in pandas DataFrame"""

        pairs = uniformly_random_pairs(self.robin_config.num_assays)
        all_dfs=[]

        for i, (idx_1,idx_2) in enumerate(pairs):
            logger.info(f"Running comparison {i + 1} for pair ({idx_1}, {idx_2})")
            try:
                result = self.compare_single_pair(idx_1, idx_2)
                # store each result for further study
                output_path = self.file_system.get_individual_ranker_comparison_path(comparison_index=i)

                with open(output_path, 'w') as f:
                    json.dump(crew_output_to_json(result), f, indent=2)

                df = crew_output_to_dataframe(result)
                df["pair_index"] = i
                df["pair"] = [(idx_1, idx_2)] * len(df)
                all_dfs.append(df)
                logger.success(f"Comparison {i + 1} succeeded!")
            except Exception as e:
                logger.error(f"Comparison {i + 1} failed: {str(e)}")

        self.ranker_result = pd.concat(all_dfs, ignore_index=True)

        return self.ranker_result

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the ranking results to a DataFrame."""
        if self.ranker_result is None:
            raise ValueError("No ranking results found. Call `.run_all_comparison()` first.")
        return self.ranker_result

    def save_df(self):
        if self.ranker_result is None:
            raise ValueError("No ranking results found. Call `.run_all_comparison()` first.")
        ranking_output_path = self.file_system.get_ranking_output_path()
        self.ranker_result.to_csv(ranking_output_path, index=False)

    def show_elo_ranker_results(self) -> pd.DataFrame:
        """Computes and displays ELO scores for the ranking results."""
        if self.ranker_result is None:
            raise ValueError("No ranking results found. Call `.run_all_comparison()` first.")
        elo_scores = compute_elo_scores(self.ranker_result)
        return elo_scores

    def show_btl_ranker_results(self) -> pd.DataFrame:
        """Computes and displays BTL scores for the ranking results."""
        if self.ranker_result is None:
            raise ValueError("No ranking results found. Call `.run_all_comparison()` first.")
        btl_scores = compute_btl_scores(self.ranker_result)
        return btl_scores
