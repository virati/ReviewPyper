# %%

import os

import numpy as np
import matplotlib.pyplot as plt

from revpyper.agents import reviewer

from dotenv import load_dotenv

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_KEY")


# %%
candidate_articles = [
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
keyword_list = []
# %%
agent = reviewer(LLM_KEY=GEMINI_KEY, LLM_MODEL="gemini/gemini-2.5-flash-preview-04-17")
goal = "Identify articles that are moderately likely to cover deep brain stimulation."

filtered_titles = agent.filter_titles(
    goal=goal, keyword_list=keyword_list, candidate_titles=candidate_titles
)
# %%
print(filtered_titles.confidence)

# %%
x_range = np.arange(len(candidate_titles))
confidence = filtered_titles.confidence
plt.plot(x_range, confidence)
ax = plt.xticks(x_range, labels=[a[:15] for a in candidate_titles], rotation=90)
