import dspy
from typing import List


class relevant_articles(dspy.Signature):
    goal: str = dspy.InputField(
        desc="Determine how relevant each paper is to the stated goal, and the underlying target concept, through the titles and keywords provided."
    )
    candidate_titles: List[str] = dspy.InputField(
        desc="List of candidate article titles to filter. Pay attention to how particular words, or names, in the title may relate to the goal.  Be sure to assess higher order links between words and the target concept."
    )
    keyword_list: List[str] = dspy.InputField(
        desc="List of keywords to filter titles by."
    )
    confidence: List[float] = dspy.OutputField(
        desc="Confidence score, between 0 and 100 with 100 being most confident, for all titles in the list."
    )
