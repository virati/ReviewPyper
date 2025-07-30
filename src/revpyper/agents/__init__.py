import os
from typing import List

import dspy
from dotenv import load_dotenv

from revpyper.agents.filters import relevant_articles
import numpy as np
import matplotlib.pyplot as plt


load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_KEY")


class reviewer:
    def __init__(
        self,
        LLM_MODEL: str = "gemini/gemini-2.5-flash-preview-04-17",
        topics: List[str] = None,
        paper_list: List = None,
    ):
        if "gemini" in LLM_MODEL.lower():
            LLM_KEY = GEMINI_KEY
        else:
            raise ValueError("Unsupported LLM model.")

        if topics is not None and paper_list is not None:
            raise ValueError("Please provide either topics or paper_list, not both.")

        self._topics = topics
        self._paper_list = paper_list

        self.lm = dspy.LM(LLM_MODEL, api_key=LLM_KEY, max_tokens=6000)
        dspy.configure(lm=self.lm)

    def set_goal(self, goal: str):
        # a goal is a meta-topic, we'll let an LLM come in and translate a goal into a set of topics
        self._goal = goal
        # ask the LLM to translate a goal into a set of topics here

    def filter_titles(
        self, goal: str, keyword_list: List[str], candidate_titles: List[str]
    ) -> List[str]:
        filter_process = dspy.Predict(relevant_articles)

        response = filter_process(
            goal=goal, candidate_titles=candidate_titles, keyword_list=keyword_list
        )
        return response

    def import_papers(self, import_structure=None):
        if import_structure is None:
            logging.info("No import structure provided, .")
