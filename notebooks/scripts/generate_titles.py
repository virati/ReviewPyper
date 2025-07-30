# %%
# this script will show how to generate a list of titles for a particular topic
# We'll take several approaches: One based on LLM and One based on Estiblished APIs/Keyword
verbatim_topic = ""
distilled_topics = {}


# %%

from revpyper.agents import reviewer

agent = reviewer(
    base_models={"all": "gemini(any)"}
)  # key should be pulled automatically from .env


paper_list = agent.pull_papers(topics=distilled_topics)

paper_titles = [paper["title"] for paper in paper_list]
