## ChatBloom


## Install
```bash
conda create -n coati
conda activate coati
pip install .

cd ..
git clone https://github.com/hpcaitech/transformers
cd transformers
pip install .

cd ..
git clone git@github.com:hpcaitech/ColossalAI.git
cd ColossalAI
git checkout e6a132a
pip install .
```

## Training
### Instruction Tuning (Optional)
This is optional, because we can use [bloomz](https://huggingface.co/bigscience/bloomz-1b7) directly.

Data:
<!-- - [pCLUE](https://huggingface.co/datasets/wbbbbb/pclue) | [github](https://github.com/CLUEbenchmark/pCLUE)
- [BELLE Generated Chat](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)
- [BELLE train_2M_CN](https://huggingface.co/datasets/BelleGroup/train_2M_CN) -->

|Dataset | Size | Used |
| - | - | - |
| [pCLUE](https://huggingface.co/datasets/wbbbbb/pclue) | 1.2M | 0.3M |
| [BELLE Generated Chat](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M) | 0.4M | 0.2M |
| [BELLE train_2M_CN](https://huggingface.co/datasets/BelleGroup/train_2M_CN) | 2M | 0.5M |


Run:
```bash
bash scripts/train_instruction_tuning.sh
```

### SFT

Data:

|Dataset | Size | Used |
| - | - | - |
| [BELLE/1M](https://huggingface.co/datasets/BelleGroup/train_1M_CN) | 1M | 0.1M |
| [BELLE Multiturn Chat](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M) | 0.8M | 0.2M |
| [InstructionWild](https://github.com/XueFuzhao/InstructionWild) | 52k * 2 | 30k * 2 |

format as
```
Human: [Instruction or Input]
Assistant: [Output]
```

Run:
```bash
bash scripts/train_sft.sh
```

### RM
TODO
```bash
bash scripts/train_rm.sh
```

### PPO-RL 
TODO
```bash
bash scripts/train_ppo.sh
```

## Limitation and Usage Limits
The datasets we used (e.g. [BELLE](https://github.com/LianjiaTech/BELLE)) require developers only use the data for research purposes. Thus, commercial and other potentially harmful uses of our models are not allowed.

## Acknowledgements
This project is based on [ColossalAI](https://github.com/hpcaitech/ColossalAI) and [ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat), and thanks to these projects for their contributions to open source.


