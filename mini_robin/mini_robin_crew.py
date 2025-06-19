import os
from dotenv import load_dotenv
from typing import List
from crewai import LLM
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import RagTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from .file_system import FileSystem

load_dotenv(dotenv_path="/Users/sarangsonar/Documents/GitHub/mini_robin/mini_robin/.env") # EmbedChain is using OpenAI API key, so we need to load it from .env file


@CrewBase
class MiniRobinAssayCrew:
    """
    Phase 1: Generates hypotheses and saves to JSON
    """

    def __init__(self, llm: LLM,
                 files: List[str],
                 file_system: FileSystem,
                 ):
        self.llm = llm
        self.files = files
        self.file_system = file_system
        self.rag_tool = RagTool()
        for file_path in self.files:
            self.rag_tool.add(source=file_path, data_type="pdf_file")

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/hypotheses_generation_crew/hypotheses_agents.yaml" # TODO: Dont hardcode this, pass argument
    tasks_config = "config/hypotheses_generation_crew/hypotheses_tasks.yaml"


    @agent
    def literature_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['literature_researcher'],
            verbose=True,
            memory=True,
            llm=self.llm
        )

    @agent
    def assay_proposer(self) -> Agent:
        return Agent(
            config=self.agents_config['assay_proposer'],
            verbose=True,
            memory=True,
            tools=[self.rag_tool]
        )

    @agent
    def hypothesis_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['hypothesis_writer'],
            verbose=True,
            memory=True,
            llm=self.llm
        )

    @task
    def generate_queries_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_queries_task']
        )

    @task
    def propose_assays_task(self) -> Task:
        return Task(
            config=self.tasks_config['propose_assays_task'],
            context=[self.generate_queries_task()]
        )

    @task
    def hypotheses_task(self) -> Task:
        return Task(
            config=self.tasks_config['hypotheses_task'],
            context=[self.propose_assays_task()],
            output_file=self.file_system.get_hypotheses_output_path_json()
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.literature_researcher(),
                self.assay_proposer(),
                self.hypothesis_writer(),
            ],
            tasks=[
                self.generate_queries_task(),
                self.propose_assays_task(),
                self.hypotheses_task(),
            ],
            process=Process.sequential,
            verbose=False
        )


@CrewBase
class MiniRobinRankingCrew:
    """
    Phase 2: Loads hypotheses, does pairwise comparisons, ranks them
    """
    def __init__(self, llm: LLM):
        self.llm = llm

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/ranking_crew/ranking_agent.yaml"
    tasks_config = "config/ranking_crew/ranking_task.yaml"

    @agent
    def assay_critic_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['assay_critic_expert'],
            verbose=True,
            memory=True,
            llm=self.llm
        )

    @task
    def ranking_task(self) -> Task:


        return Task(
            config=self.tasks_config['ranking_task'],
            # TODO: check if there is a need for storing the immediate output of the individual ranking comparsions
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.assay_critic_expert(),
            ],
            tasks=[
                self.ranking_task(),
            ],
            process=Process.sequential,
            verbose=False
        )


@CrewBase
class MiniRobinGoalCrew:
    """
    Phase 3: Based on the ranking output and selected assay hypotheses, this crew is responsible for
    synthesizing a concise and specific research goal for the *next* stage of identifying therapeutic compounds
    to test this assay.
    """

    def __init__(self, llm: LLM,
                 crew_config_directory: str = "config/goal_synthesis_crew"):
        self.llm = llm

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/goal_synthesis_crew/goal_synthesis_agent.yaml"
    tasks_config = "config/goal_synthesis_crew/goal_synthesis_task.yaml"

    @agent
    def goal_synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config['goal_synthesizer'],
            verbose=True,
            memory=True,
            llm=self.llm
        )

    @task
    def synthesize_goal_task(self) -> Task:
        return Task(
            config=self.tasks_config['synthesize_goal_task'],
            context=[self.ranking_task()]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.goal_synthesizer(),
            ],
            tasks=[
                self.synthesize_goal_task(),
            ],
            process=Process.sequential,
            verbose=False
        )




