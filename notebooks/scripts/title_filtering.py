# %%
from dotenv import load_dotenv
import os
import dspy

from typing import List
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
candidate_titles: List[str] = [
    "Parachute use to prevent death and major trauma related to gravitational challenge: systematic review of randomised controlled trials.",
    "Invasive Fungal Disease Complicating Coronavirus Disease 2019: When It Rains, It Spores.",
    "Vancomycin and the Risk of AKI: Now Clearer than Mississippi Mud.",
    "Hitting the target with non-invasive deep brain stimulation: Potential therapy for addiction, depression, and OCD.",
    "Clinical use of the polymyxins: the tale of the fox and the cat.",
    "Deep brain stimulation: current challenges and future directions.",
    "Bundle in the Bronx: Impact of a Transition-of-Care Outpatient Parenteral Antibiotic Therapy Bundle on All-Cause 30-Day Hospital Readmissions.",
    "Mount Sinai Is First in the Nation to Perform Deep Brain Stimulation Implant as Part of Clinical Trial for Depression.",
    "Cryptococcus neoformans: the yeast that likes it hot.",
    "Researchers use deep brain stimulation to map therapeutic targets for four brain disorders.",
    "Getting to the bottom of anal evolution.",
    "Randomized clinical trial of deep brain stimulation for poststroke pain.",
    "Electronic Health Records and the Increasing Complexity of Medical Practice: “It Never Gets Easier, You Just Go Faster”.",
    "Deep brain stimulation as an effective treatment option for post–midbrain infarction-related tremor as it presents with Benedikt syndrome.",
    "Salmonella excretion in joy-riding pigs.",
    "Do not snog the dog: infective endocarditis due to Capnocytophaga canimorsus.",
    "Fantastic yeasts and where to find them: the hidden diversity of dimorphic fungal pathogens.",
    "Experimental replication shows knives manufactured from frozen human feces do not work.",
    "Hogwarts Headaches — Misery for Muggles.",
    "Everything is awesome: Don’t forget the Lego.",
    "Transcranial Magnetic Stimulation: A Made Up Review Article",
    "Noninvasive Brain Stimulation: A Focused, Still Made Up, Review",
    "Helen Mayberg's Contribution to Neuropsychiatry: A Made Up Review",
    "A Life of Circuits: Mahlon DeLong's Story",
    "The Lord of the Rings",
    "Severance: The Screenplay",
    "Linear Algebra for Engineers",
]
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
