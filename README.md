# HeTGB

The code and datasets used for our paper [*"A Comprehensive Benchmark for Heterophilic Text-Attributed Graphs"*](https://arxiv.org/abs/2503.04822).

## Data

You can find both the raw and processed datasets at the following link on Hugging Face: [HeTGB](https://huggingface.co/datasets/0219shujie/HeTGB).

Each dataset is in `.npz` format and includes:
`edges`, `node_labels`, `node_features`, `node_texts`, `label_texts`, `train_masks`, `val_masks`, `test_masks`.

The data preprocessing process can be referred to in the `data_process` folder.

## Reproduce the Results for LLM4HeG

### Stage1

Set up the environment following the `README.md` files in the `src/LLM` directories.

#### Finetune Vicuna 7B

Generate training and inference data using`util/generate_prompt_json.py`

Fine-tune the Vicuna 7B model, merge it with any additional components, and then obtain the inference results:
```bash
cd src/LLM
bash scripts/train_lora.sh
python fastchat/model/apply_lora.py
bash eval.sh
```
Convert the inference results to the Stage 2 data format with`util/llm_result.py`

### Stage2

#### Installation

Install the required packages using the dependencies listed in `requirements.txt`.

#### Run

To run experiments, execute the `run_train.sh` script located in the `src/GNN` directory.

To compute averaged statistics, use the `src/parse_results.py` script with the `--result_path` argument, providing the path to your experiment results.

## Reproducing Other Baseline Results

To reproduce the results of other baselines, you can refer to the following implementations:

- **GCN**: [GCN GitHub Repository](https://github.com/tkipf/gcn)
- **GAT**: [GAT GitHub Repository](https://github.com/Diego999/pyGAT)
- **GraphSAGE**: [GraphSAGE GitHub Repository](https://github.com/twjiang/graphSAGE-pytorch)
- **H2GCN**: [H2GCN GitHub Repository](https://github.com/GemsLab/H2GCN)
- **FAGCN, JacobiConv**: [Heterophily Specific Models GitHub Repository](https://github.com/heterophily-submit/HeterophilySpecificModels)
- **GBK-GNN**: [GBK-GNN GitHub Repository](https://github.com/xzh0u/gbk-gnn)
- **OGNN**: [OGNN GitHub Repository](https://github.com/LUMIA-Group/OrderedGNN)
- **SEGSL**: [SE-GSL GitHub Repository](https://github.com/ringbdstack/se-gsl)
- **DisamGCL**: [Disambiguated GNN GitHub Repository](https://github.com/tianxiangzhao/disambiguatedgnn)
- **G2P2**: [G2P2 GitHub Repository](https://github.com/WenZhihao666/G2P2)
- **GraphGPT**: [GraphGPT GitHub Repository](https://github.com/HKUDS/GraphGPT)
- **LLaGA**: [LLaGA GitHub Repository](https://github.com/VITA-Group/LLaGA)
- **LLM4HeG**: [LLM4HeG GitHub Repository](https://github.com/honey0219/LLM4HeG)

## Fine-Tuning Vicuna 7B, Bloom560M, and Bloom1B1

To fine-tune Vicuna 7B, Bloom560M, and Bloom1B1 models, you can refer to the following repositories for detailed instructions:

- **FastChat**: [FastChat GitHub Repository](https://github.com/lm-sys/FastChat)
- **LLaMA-Factory**: [LLaMA-Factory GitHub Repository](https://github.com/hiyouga/LLaMA-Factory)



