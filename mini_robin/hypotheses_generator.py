import pandas as pd

from .mini_robin_crew import MiniRobinAssayCrew
from .configuration import RobinConfiguration
from .utils import load_hypotheses_json_to_df


class HypothesesGenerator:
    """
    Runs Phase 1 of the MiniRobin pipeline to generate assay hypotheses.
    """

    def __init__(self, robin_config: RobinConfiguration,
                 generator_assay_crew_instance=MiniRobinAssayCrew
                 ):
        """
        Parameters:
        - config (dict): Input config with 'topic', 'num_queries', 'num_assays', 'num_candidates'
        - llm (LLM): Initialized LLM client
        - crew_runner_cls (Callable): Crew class to use for hypothesis generation
        """
        self.robin_config = robin_config
        self.generator_assay_crew_instance = generator_assay_crew_instance
        self.result = None

    def generate(self) -> dict:
        """
        Runs the hypothesis generation phase.

        Returns:
        - result (dict): Output from crew.kickoff()
        """
        inputs = {
            "topic": self.robin_config.disease_name,
            "num_queries": self.robin_config.num_queries,
            "num_assays": self.robin_config.num_assays,
            "num_candidates": self.robin_config.num_candidates,
        }
        crew = self.generator_assay_crew_instance.crew()
        self.result = crew.kickoff(inputs=inputs)
        return self.result

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the generated hypotheses to a DataFrame.

        Returns:
        - DataFrame with hypothesis information
        """
        if self.result is None:
            raise ValueError("No result found. Call `.generate()` first.")

        # TODO: not the best design here, but its a temporary fix
        file_system = self.generator_assay_crew_instance.file_system
        hypotheses_path = file_system.get_hypotheses_output_path_json()

        return load_hypotheses_json_to_df(hypotheses_path)

    def save_df(self):
        """
        Saves the generated hypotheses DataFrame to a file.
        """
        df = self.to_dataframe()
        file_system = self.generator_assay_crew_instance.file_system
        output_path = file_system.get_hypotheses_output_path_csv()
        df.to_csv(output_path, index=False)
        print(f"Hypotheses saved to {output_path}")
