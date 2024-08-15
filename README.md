# RaDIO: Real-Time Hallucination Detection with Contextual Index Optimized Query Formulation for Dynamic Retrieval Augmented Generation

---

🚀 **Exciting News**! 

✨ We are **thrilled** to announce that our paper, titled **"RaDIO: Real-Time Hallucination Detection with Contextual Index Optimized Query Formulation for Dynamic Retrieval Augmented Generation"**, has been **accepted** for presentation at **AAAI 2025**! 🎉
🎉
---


# Code



## Install environment

```bash
conda create -n RaDIO python=3.9
conda activate RaDIO
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run RaDIO


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
python ./src/fix_csv.py
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

# Citation

```
@inproceedings{zhu2025radio,
  title={RaDIO: Real-Time Hallucination Detection with Contextual Index Optimized Query Formulation for Dynamic Retrieval Augmented Generation},
  author={Zhu, Jia and Guo, Hanghui and Shi, Weijie and Chen, Zhangze and De Meo, Pasquale},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={24},
  pages={26129--26137},
  year={2025}
}
```

