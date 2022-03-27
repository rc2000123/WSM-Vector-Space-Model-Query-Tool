# WSM-Vector-Space-Model-Query-Tool

Query Tool for Information Retrieval using a Vector Space Model. Developed by Ryan Chu 107703035 at National Cheng Chi University 

# TL:DR Example
```sh
python VectorSpace.py --query Trump Biden Taiwan China --option 1 --source EnglishNews/
```

# Parameters:
  ## --option:
    - (1) Term Frequency (TF) Weighting + Cosine Similarity
    - (2) Term Frequency (TF) Weighting + Euclidean Distance
    - (3) TF-IDF Weighting + Cosine Similarity
    - (4) TF-IDF Weighting + Euclidiean Distance
    - (5) Term Frequency (TF-IDF) Weighting + Cosine Similarity with Feedback
  ## --source:
    - define the document folder source:
      - EnglishNews/ : 7,034 English News collected from reuters.com.
      - News/ : a set of 1,500 News (1000 Chinese news and 500 English news) collected from chinatimes.com and setn.com
  ## --query:
    string used for query (please use a Chinese News source if query is in Chinese)
    
# Third Party Tools:
  numpy,nltk(English NLP processing),jieba(Chinese NLP processing),tqdm
  python version: 3.8.8
