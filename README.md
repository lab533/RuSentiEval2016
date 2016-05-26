Sentiment Russian Texts
==============================

Sentiment Analysis for Dialogue Evaluation 2016 Using Syntax Tree and Convolution Neural Net


To quick start use **notebooks**



Project Organization
------------

├── data
│   ├── interim
│   │   ├── ancestors.txt
│   │   ├── labels.txt
│   │   ├── sentences.txt
│   │   ├── siblings.txt
│   │   └── token2id.txt
│   ├── processed
│   │   └── SentiRuEval2016.pickle
│   └── raw
├── LICENSE
├── models
│   ├── architecture.json
│   ├── bank_bank_news_vec22_weights.h5
│   ├── bank_bank_random_vec22_weights.h5
│   ├── bank_bank_twitter_vec22_weights.h5
│   ├── common_bank_news_vec22_weights.h5
│   ├── common_bank_random_vec22_weights.h5
│   ├── common_bank_twitter_vec22_weights.h5
│   ├── common_ttk_news_vec22_weights.h5
│   ├── common_ttk_random_vec22_weights.h5
│   ├── common_ttk_twitter_vec22_weights.h5
│   ├── ttk_ttk_news_vec22_weights.h5
│   ├── ttk_ttk_random_vec22_weights.h5
│   └── ttk_ttk_twitter_vec22_weights.h5
├── notebooks
│   ├── train_model.ipynb
│   ├── transform_data.ipynb
│   └── work_with_model.ipynb
├── README.md
├── src
│   ├── data
│   │   └── make_dataset.py
│   ├── features
│   │   ├── feature_proc.py
│   │   └── __pycache__
│   │       ├── build_features.cpython-34.pyc
│   │       └── feature_proc.cpython-34.pyc
│   ├── __init__.py
│   └── model
│       ├── predict_model.py
│       ├── __pycache__
│       │   └── train_model.cpython-34.pyc
│       └── train_model.py
└── User.Interests.txt

