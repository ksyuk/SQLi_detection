# SQL injection attack detection by Deep Leaning and LLM

Creating SQL injection attack detection models using CNN, BiLSTM, and BERT. Also prompt engineering to detect attack using GPT3.5, GPT4, Llama2-7B, and Mistral-7B.

## Prerequiresed for LLM
### GPT
```
Setting up Open AI API
```

### Llama, Mistral
```
Hugging Face account
Access right to Llama2
High-performance GPU
```

## Dataset  
1. Download dataset in kaggle.  
https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset  

2. Place the folder named `dataset` in project root.  

3. Rename `sqli.csv` to `sqliv1.csv`, `SQLiV3.csv` to `sqliv3.csv`.  

4. Set all files to utf-8 character encoding.  

5. Edit the dataset by using files in `dataset/edit_dataset.py`.

## Virtual Environment
```zsh
python3 -m venv venv
```

```zsh
(Linux/Mac)
source venv/bin/activate
```

```zsh
pip install -r requirements.txt
```
