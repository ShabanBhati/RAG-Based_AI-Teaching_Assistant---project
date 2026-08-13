# import requests
# # import os
# # import json
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from read_chunks import create_embedding
# import joblib

# df = joblib.load('embeddings.joblib')

# def create_embedding(text_list):
#     # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })

#     response_json = r.json()

#     if "embeddings" not in response_json:
#         print("OLLAMA RESPONSE (error dekhne ke liye):", response_json)
#         raise RuntimeError("Embeddings nahi mile — upar wala response dekho.")

#     return response_json["embeddings"]


# jsons = os.listdir("jsons")  # List all the jsons
# my_dicts = []
# chunk_id = 0

# for json_file in jsons:
#     with open(f"jsons/{json_file}") as f:
#         content = json.load(f)
#     print(f"Creating Embeddings for {json_file}")
#     embeddings = create_embedding([c['text'] for c in content['chunks']])

#     for i, chunk in enumerate(content['chunks']):
#         chunk['chunk_id'] = chunk_id
#         chunk['embedding'] = embeddings[i]
#         chunk_id += 1
#         my_dicts.append(chunk)
#     break

# df = pd.DataFrame.from_records(my_dicts)
# #save this dataframe
# joblib.dump(df, 'embeddings.joblib')
 
# incoming_query = input("Ask a Question: ")
# question_embedding = create_embedding([incoming_query])[0]

# similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
# top_results = 3
# max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
# new_df = df.loc[max_indx]
# print(new_df[["title", "number", "text"]])


# import pandas as pd 
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np 
# import joblib 
# import requests


# def create_embedding(text_list):
#     # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })

#     embedding = r.json()["embeddings"] 
#     return embedding


# df = joblib.load('embeddings.joblib')


# incoming_query = input("Ask a Question: ")
# question_embedding = create_embedding([incoming_query])[0] 

# # Find similarities of question_embedding with other embeddings
# # print(np.vstack(df['embedding'].values))
# # print(np.vstack(df['embedding']).shape)
# similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
# top_results = 3
# max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
# new_df = df.loc[max_indx] 
# print(new_df[["title", "number", "text"]])


# import pandas as pd 
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np 
# import joblib 
# import requests


# def create_embedding(text_list):
#     # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })

#     embedding = r.json()["embeddings"] 
#     return embedding


# df = joblib.load('embeddings.joblib')


# incoming_query = input("Ask a Question: ")
# question_embedding = create_embedding([incoming_query])[0] 

# # Find similarities of question_embedding with other embeddings
# # print(np.vstack(df['embedding'].values))
# # print(np.vstack(df['embedding']).shape)
# similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# # print(similarities)
# top_results = 30
# max_indx = similarities.argsort()[::-1][0:top_results]
# # print(max_indx)
# new_df = df.loc[max_indx] 
# # print(new_df[["title", "number", "text"]])

# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])



# import pandas as pd 
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np 
# import joblib 
# import requests


# def create_embedding(text_list):
#     # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })

#     embedding = r.json()["embeddings"] 
#     return embedding

# def inference(prompt):
#     r = requests.post("http://localhost:11434/api/generate", json={
#         # "model": "deepseek-r1",
#         "model": "llama3.2",
#         "prompt": prompt,
#         "stream": False
#     })

#     response = r.json()
#     print(response)
#     return response

# df = joblib.load('embeddings.joblib')


# incoming_query = input("Ask a Question: ")
# question_embedding = create_embedding([incoming_query])[0] 

# # Find similarities of question_embedding with other embeddings
# # print(np.vstack(df['embedding'].values))
# # print(np.vstack(df['embedding']).shape)
# similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# # print(similarities)
# top_results = 5
# max_indx = similarities.argsort()[::-1][0:top_results]
# # print(max_indx)
# new_df = df.loc[max_indx] 
# # print(new_df[["title", "number", "text"]])

# prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

# {new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
# ---------------------------------
# "{incoming_query}"
# User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
# '''
# with open("prompt.txt", "w") as f:
#     f.write(prompt)

# response = inference(prompt)["response"]
# print(response)

# with open("response.txt", "w") as f:
#     f.write(response)
# # for index, item in new_df.iterrows():
# #     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])





import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests


def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    print(response)
    return response

df = joblib.load('embeddings.joblib')


incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0] 

# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx] 
# print(new_df[["title", "number", "text"]])

prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)
# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])
