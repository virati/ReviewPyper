import dspy
import os

from dotenv import load_dotenv

load_dotenv()


class agent:
    def __init__(self, base_model="gemini/gemini-2.5-flash-preview-04-17"):
        if "gemini" in base_model.lower():
            LLM_KEY = os.getenv("GEMINI_KEY")

        self.lm = dspy.LM(base_model, api_key=LLM_KEY, max_tokens=6000)
        dspy.configure(lm=self.lm)
