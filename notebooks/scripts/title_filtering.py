# %%
from dotenv import load_dotenv
import os
import dspy

from typing import List, Tuple
import entrezpy.esearch.esearcher
import entrezpy.esearch.esearch_analyzer
import pandas as pd


load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_KEY")


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


# %%
class reviewer:
    def __init__(self):
        lm = dspy.LM(
            "gemini/gemini-2.5-flash-preview-04-17", api_key=GEMINI_KEY, max_tokens=6000
        )
        dspy.configure(lm=lm)

    def filter_titles(
        self, goal: str, keyword_list: List[str], candidate_titles: List[str]
    ) -> List[str]:
        filter_process = dspy.Predict(relevant_articles)

        response = filter_process(
            goal=goal, candidate_titles=candidate_titles, keyword_list=keyword_list
        )
        return response


def pull_pubmed_data(return_field: List[str] = ["title", "abstract", "authors"]):
    Entrez.email = os.getenv("ENTREZ_EMAIL")
    handle = Entrez.esearch(db="pubmed", term="deep brain stimulation", retmax=100)
    record = Entrez.read(handle)
    handle.close()

    ids = record["IdList"]
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
    records = Entrez.parse(handle)

    results = []
    for record in records:
        entry = {}
        for field in return_field:
            if field == "title":
                entry[field] = record.get("TI", "")
            elif field == "abstract":
                entry[field] = record.get("AB", "")
            elif field == "authors":
                entry[field] = record.get("AU", [])
        results.append(entry)

    return pd.DataFrame(results)


# %%
candidate_articles: List[Tuple[str, bool]] = [
    (
        "Parachute use to prevent death and major trauma related to gravitational challenge: systematic review of randomised controlled trials.",
        False,
    ),
    (
        "Invasive Fungal Disease Complicating Coronavirus Disease 2019: When It Rains, It Spores.",
        False,
    ),
    (
        "Hitting the target with non-invasive deep brain stimulation: Potential therapy for addiction, depression, and OCD.",
        True,
    ),
    ("Deep brain stimulation: current challenges and future directions.", True),
    (
        "Electronic Health Records and the Increasing Complexity of Medical Practice: “It Never Gets Easier, You Just Go Faster”.",
        False,
    ),
    ("Transcranial Magnetic Stimulation: A Made Up Review Article", False),
    ("Noninvasive Brain Stimulation: A Focused, Still Made Up, Review", False),
    ("Helen Mayberg's Contribution to Neuropsychiatry: A Made Up Review", True),
]
candidate_titles = [a[0] for a in candidate_articles]
candidate_truths = [a[1] for a in candidate_articles]
keyword_list: List[str] = []
# %%
agent = reviewer()
goal = "Identify articles that are moderately likely to cover deep brain stimulation."
filtered_titles = agent.filter_titles(
    goal=goal, keyword_list=keyword_list, candidate_titles=candidate_titles
)
# %%
print(filtered_titles.confidence)

# %%

import numpy as np
import matplotlib.pyplot as plt

x_range = np.arange(len(candidate_titles))
confidence = filtered_titles.confidence
plt.plot(x_range, confidence)
ax = plt.xticks(x_range, labels=[a[:15] for a in candidate_titles], rotation=90)
