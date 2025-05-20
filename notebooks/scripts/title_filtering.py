# %% [markdown]
# # Title Filtering
# This notebook takes a list of articles (eg csv) and filters for a specific topic based on **article title**.

# %%
# bring in relevant API keys
from dotenv import dotenv_values

config = dotenv_values("../../.env")

# %% [markdown]
# ## First Assess Titles
# Define Path to CSV and Question
# - csv_path = "path/to/your/titles.csv"
# - question = "Is this title related to medical research?"
# - keywords_list = ["focal", "lesion", "brain", "death", "case"]


# paper_list_csv_path = "/Users/cu135/Partners HealthCare Dropbox/Calvin Howard/studies/ccm_memory/results/review_pyper/invasive/1962-2013_spreadsheet.csv"
# core_question = "Does this article look like it may contain a report of invasive brain stimulation altering memory?"
paper_list_csv_path = "../../assets/test_papers.csv"
system_prompt = ""
core_question_prompt = ""
examples_prompt = "Examples could include articles that talk about `brain stimulation`, `invasive EEG`, or `DBS`, and may discuss `memory enhancement`, `memory impairment`, or diseases like `alzheimers`."


# %% [markdown]
# If you also want to perform a keyword-based assessment (free), you can enter a list of strings here:
# - Just set to None if you don't want to use it. But it's a good baseline.
# - example: ["Alice in Wonderland Syndrome", "macropsia", "micropsia"]
keywords_list = None

# %%
from calvin_utils.gpt_sys_review.gpt_utils import TitleScreener

title_screening = TitleScreener(
    api_key_path=open_ai_key,
    csv_path=paper_list_csv_path,
    question=question,
    keywords=keywords_list,
    model_choice="gpt3_small",
)
title_screening.run()

# %% [markdown]
# Your titles have now been screened.
# - If you are curious about screening titles then abstracts versus titles and abstracts, please see this study:
#     - doi: 10.2147/CLEP.S43118
# - Enjoy. If this has been helpful, please consider adding Calvin Howard as a collaborator.
# - e: choward12@bwh.harvard.edu
