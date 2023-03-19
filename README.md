# Insights on the Success and Failure of Early-Stage Startups - An Application of Text Mining and Natural Language Processing

**Introduction**

This repository contains the code relating to my master thesis on "Insights on the Success and Failure of Early-Stage Startups - An Application of Text Mining and Natural Language Processing" 
written at TUM Entrepreneurship Research Institute, Technical University Munich. 

It leverages the code provided in https://github.com/dsridhar91/hstm. 

**Abstract**

This thesis explores the use of natural language processing, specifically the Heterogeneous Supervised Topic Modeling (HSTM) algorithm, to analyze interviews with entrepreneurs in order to predict the success or failure of early-stage startups. While there has been research on the factors that contribute to startup success, such as the founding team and access to resources, this paper seeks to identify whether interview data contains insights that could help predict the success or failure of startups. The study analyzes 297 interviews with founders of failed and successful startups and finds that three question categories - From Idea to Product, Marketing Strategies, and Challenges, Obstacles and Mistakes - predict startup success with good accuracy. The results demonstrate the potential of applying natural language processing techniques to uncover hidden patterns in textual data and contribute to theoretical and practical implications for new venture success in the field of entrepreneurship.

**Requirements** 

The code has been tested on Python 3.6.9 with the following packages:
```bash
dill==0.2.8.2
nltk~=3.8
numpy~=1.24.1
pandas~=1.5.2
scikit-learn~=1.2.0
scipy~=1.9.3
tensorflow==2.1.0
tensorflow-gpu==2.1.0
tensorflow-hub==0.7.0
tensorflow-probability==0.7.0
tokenizers==0.7.0
torch==1.3.1
torchvision==0.4.2
transformers~=4.25.1
bs4~=4.11.1
requests~=2.28.1
beautifulsoup4~=4.11.1
cufflinks~=0.17.3
plotly~=5.11.0
textblob~=0.15.3
sklearn~=0.0.post1
```

It is possible to install the dependencies using pip:
```bash
pip install -r requirements.txt
```

**Data**

The data set was retrieved from Failory through webscraping. Failory is a content site for startups, which publishes
interviews with founders. At the time of scraping (July 2022), the website contained 290 interviews with entrepreneurs 
of failed and successful early-stage startups. In this context, success  and failure are defined as perceived by the 
founder at the time of the interview and are not measurable by objective metrics such as funding, revenue or seed phase.

**Reproducing the Study**

0. Optional: Scrap most recent interview data with data/scraper.py. Otherwise, the downloaded interviews are under dat/startup_data_final.csv. 
1. Optional: Preprocess data sets with src/preprocessing.py to create individual data sets for the different question categories. Preprocessed data files are already saved in the folder '/dat'.
2. Run hyperparameter tuning with src/experiment/tune_hyperparamters.py. This script will run the experiment (src/experiment/run_experiment.py) for all defined hyperparameter combinations.
3. Extract hyperparameter combinations that yield best results for each question category with evaluation/process.results.py.
4. Visualize results with the Jupyter Notebook visualize_results.ipynb.

**References**
- [1] Sridhar, D., Daumé, H., & Blei, D. (2022). _Heterogeneous Supervised Topic Models_. Transactions of the Association for Computational Linguistics, 10, 732-745.
- [2] Mcauliffe, J., & Blei, D. (2007). _Supervised topic models_. Advances in neural information processing systems, 20.






