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
### Instruction Tuning

Data:
- [pClue](https://huggingface.co/datasets/wbbbbb/pclue) | [github](https://github.com/CLUEbenchmark/pCLUE)
- [BELLE Generated Chat](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)
- [BELLE train_2M_CN](https://huggingface.co/datasets/BelleGroup/train_2M_CN)

Run:
```bash
bash scripts/train_instruction_tuning.sh
```

### SFT
- [BELLE/1M](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
- [BELLE Multiturn Chat](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)
- [InstructionWild](https://github.com/XueFuzhao/InstructionWild)

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


