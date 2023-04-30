# Data
## ShareGPT
We use the data from https://huggingface.co/datasets/jeffwan/sharegpt_vicuna and follow the script it has written to process.

```bash
git clone https://huggingface.co/datasets/jeffwan/sharegpt_vicuna
cd sharegpt_vicuna
pip install -r requirements.txt

python clean_sharegpt.py --in sharegpt_20230401_html.json --out sharegpt_20230401_clean.json

python optional_clean.py --in sharegpt_20230401_clean.json --out sharegpt_20230401_clean_lang_zh.json --lang zh
python optional_clean.py --in sharegpt_20230401_clean.json --out sharegpt_20230401_clean_lang_en.json --lang en

python merge.py sharegpt_20230401_clean_lang_zh.json sharegpt_20230401_clean_lang_en.json sharegpt_20230401_clean_lang.json

python split_long_conversation.py --in sharegpt_20230401_clean_lang.json --out sharegpt_20230401_clean_lang_split2.json --model-name bigscience/bloom-1b7 --max-length 800
```
