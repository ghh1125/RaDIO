# RaDIO: Real-Time Hallucination Detection with Contextual Index Optimized Query Formulation for Dynamic Retrieval Augmented Generation

---

🚀 **Exciting News**! 

✨ We are **thrilled** to announce that our paper, titled **"RaDIO: Real-Time Hallucination Detection with Contextual Index Optimized Query Formulation for Dynamic Retrieval Augmented Generation"**, has been **accepted** for presentation at **AAAI 2025**! 🎉
🎉
---


## Contextual Index Optimized Query Formulation



## Install environment

```bash
conda create -n RaDIO python=3.9
conda activate RaDIO
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run RaDIO


### MLP train

```
python generate_data.py
```
and
```
python ./scr/MLP_train/MLP.py
```



### Build index


```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

```bash
cd data
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-7.17.9.tar.gz
rm elasticsearch-7.17.9.tar.gz 
cd elasticsearch-7.17.9
nohup bin/elasticsearch &  # run Elasticsearch in background
cd ../..
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki  # build index
```

### Download Dataset

For 2WikiMultihopQA:

Download the [2WikiMultihop](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1) dataset from its repository <https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1>. Unzip it and move the folder to `data/2wikimultihopqa`.

For StrategyQA:

```bash
wget -O data/strategyqa_dataset.zip https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip
mkdir -p data/strategyqa
unzip data/strategyqa_dataset.zip -d data/strategyqa
rm data/strategyqa_dataset.zip 
```

For HotpotQA:

```bash
mkdir -p data/hotpotqa
wget -O data/hotpotqa/hotpotqa-dev.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

For IIRC:

```bash
wget -O data/iirc.tgz https://iirc-dataset.s3.us-west-2.amazonaws.com/iirc_train_dev.tgz
tar -xzvf data/iirc.tgz
mv iirc_train_dev/ data/iirc
rm data/iirc.tgz
```

### Download SBERT,SGPT

```
./huggingface/paraphrase-MiniLM-L6-v2
```
and
```
./hugggingface/SGPT
```
### Fix  psgs_w100.tsv

```
python ./src/jiancha.py
```

### Run

```
bash train.sh
```

##### baseline : ./src/oldgenerate.py
#### Two calculation methods
##### RaDIO: ./src/generate.py
##### RaDIO: ./src/new_generate.py

### Evaluate


```bash
python ./src/evaluate1.py --dir path_to_folder(result/2wikimultihopqa_llama2_7b/newBM25)
```


# Appendix

In this experiment, we conduct a comprehensive case study to compare the performance of the Base and RaDIO methods. Our analysis focused on evaluating the predictions and associated metrics (Hc, Sc, and Gc) for each case, utilizing five representative cases to illustrate the effectiveness of these methods in multi-hop reasoning tasks.

### Case 1: Does Santa Claus work during summer?

- **Base method prediction:**  
  The text states that Howard worked at the train station, which is a place of business. Businesses are typically open during the summer. Thus, Santa Claus works during summer. So the answer is yes.  
  - **Outcome:** ❌  
  - **Hc:** 2 | **Sc:** 16 | **Gc:** 5

- **RaDIO method prediction:**  
  Santa Claus works during winter. Winter is from December to February. Summer is from June to August. Thus, Santa Claus does not work during summer. So the answer is no.  
  - **Outcome:** ✅  
  - **Hc:** 1 | **Sc:** 9 | **Gc:** 3

### Case 2: Would an uninsured person be more likely than an insured person to decline a CT scan?

- **Base method prediction:**  
  An uninsured person would be more likely than an insured person to decline a CT scan. Reasoning:  
  1. The diagnostic accuracy of coronary CT angiography (CCTA) is similar to that of invasive coronary angiography via cardiac catheterization (ICA).  
  2. Both CCTA and ICA yield similar prognostic information, such as stenosis severity and extent of coronary artery disease.  
  - **Outcome:** ❌  
  - **Hc:** 4 | **Sc:** 46 | **Gc:** 8

- **RaDIO method prediction:**  
  An uninsured person may have financial constraints that would make them less likely to get a CT scan. In contrast, an insured person may have greater access to healthcare resources, including CT scans. Thus, an uninsured person may be more likely to decline a CT scan than an insured person. So the answer is yes.  
  - **Outcome:** ✅  
  - **Hc:** 2 | **Sc:** 11 | **Gc:** 5

### Case 3: Are Christmas trees dissimilar to deciduous trees?

- **Base method prediction:**  
  1. The text states that the Christmas tree was a 40-foot Colorado Blue Spruce donated by the Loving Shepherd Lutheran Church and School.  
  2. Deciduous trees are trees that lose their leaves in the fall.  
  3. Colorado Blue Spruce trees are evergreen, which means they keep their needles year-round. Thus, the Christmas tree is not dissimilar to deciduous trees. So the answer is no.  
  - **Outcome:** ❌  
  - **Hc:** 6 | **Sc:** 59 | **Gc:** 13

- **RaDIO method prediction:**  
  Deciduous trees shed their leaves in the fall. Christmas trees are evergreen, meaning they keep their leaves year-round. Thus, Christmas trees are different from deciduous trees. So the answer is yes.  
  - **Outcome:** ✅  
  - **Hc:** 1 | **Sc:** 7 | **Gc:** 3

### Case 4: Is the language used in Saint Vincent and the Grenadines rooted in English?

- **Base method prediction:**  
  The passage states that the official language of Guernsey is English, but it also mentions that French was the official language until 1948. This implies that the language used in Guernsey is not rooted in English. Therefore, the answer is no.  
  - **Outcome:** ❌  
  - **Hc:** 2 | **Sc:** 15 | **Gc:** 5

- **RaDIO method prediction:**  
  Saint Vincent and the Grenadines is an English-speaking country. English is the official language. Thus, the language used in Saint Vincent and the Grenadines is rooted in English. So the answer is yes.  
  - **Outcome:** ✅  
  - **Hc:** 1 | **Sc:** 7 | **Gc:** 3

### Case 5: Are more people today related to Genghis Khan than Julius Caesar?

- **Base method prediction:**  
  Genghis Khan lived in the 13th century. Julius Caesar lived in the 1st century. Today is in the 21st century. The distance between the 13th century and the 21st century is 890 years. There are 890 years between the number of people alive today and Genghis Khan. Thus, more people today are related to Genghis Khan than Julius Caesar. So the answer is yes.  
  - **Outcome:** ✅  
  - **Hc:** 5 | **Sc:** 33 | **Gc:** 11

- **RaDIO method prediction:**  
  Genghis Khan lived in the 13th century. Julius Caesar lived in the 1st century. Thus, there are more people alive today than there were in the 1st century. However, there are not many people alive today who are directly related to Genghis Khan. Therefore, more people today are related to Julius Caesar than Genghis Khan. So the answer is yes.  
  - **Outcome:** ✅  
  - **Hc:** 2 | **Sc:** 13 | **Gc:** 5

## Summary

In summary, the RaDIO method consistently outperformed the Base method across all cases, demonstrating significant improvements in reducing Hc, Sc, and Gc scores. This consistent performance underscores RaDIO's superior accuracy and efficiency in multi-hop reasoning tasks. The results affirm the advantages of the RaDIO method, particularly in complex logic and reasoning tasks, where its ability to deliver accurate and computationally efficient solutions provides a clear competitive edge.

