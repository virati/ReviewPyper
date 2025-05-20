#%%
import dspy
lm = dspy.LM('ollama_chat/granite3.1-dense:8b', api_base="http://localhost:11434", api_key='')
dspy.configure(lm=lm)
response = lm(messages=[{'role': 'user', 'content':'Which model are you?'}])
print(response)
#%%
def evaluate_math(expression :str ):
    return dspy.PythonInterpreter({}).execute(expression)

def search_wikipedia(query: str):
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

react = dspy.ReAct("question -> answer: float", tools = [evaluate_math, search_wikipedia])

pred = react(question="What is 2330202 divided by the year of birth of Albert Einstein?")
print(pred.answer)
#%%

