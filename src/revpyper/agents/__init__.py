import dspy
from revpyper.agents.filters import relevant_articles
from typing import List


class reviewer:
    def __init__(
        self,
        LLM_KEY: str = None,
        LLM_MODEL: str = "gemini/gemini-2.5-flash-preview-04-17",
    ):
        if LLM_KEY is None:
            raise ValueError("LLM_KEY must be provided to initialize the reviewer.")
        lm = dspy.LM(LLM_MODEL, api_key=LLM_KEY, max_tokens=6000)
        dspy.configure(lm=lm)

    def filter_titles(
        self, goal: str, keyword_list: List[str], candidate_titles: List[str]
    ) -> List[str]:
        filter_process = dspy.Predict(relevant_articles)

        response = filter_process(
            goal=goal, candidate_titles=candidate_titles, keyword_list=keyword_list
        )
        return response
