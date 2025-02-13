<div align="center">

  <h2><b>  Beam Prediction based on Large Language Models </b></h2>
</div>

<div align="center">



</div>






## Introduction
This project is for the simulation in the paper "[Beam Prediction based on Large Language Models][link2letter]". In this paper, we use large language models (LLMs) to develop a high-performing and robust beam prediction method. We formulate the millimeter wave (mmWave) beam prediction problem as a time series forecasting task, where the historical observations are aggregated through cross-variable attention and then transformed into text-based representations using a trainable tokenizer. By leveraging the prompt-as-prefix (PaP) technique for contextual enrichment, our method harnesses the power of LLMs to predict future optimal beams. 

[link2letter]:<https://arxiv.org/pdf/2408.08707>


## Requirements
- accelerate==0.20.3
- einops==0.7.0
- matplotlib==3.7.0
- numpy==1.23.5
- pandas==1.5.3
- scikit_learn==1.2.2
- scipy==1.5.4
- torch==2.0.1
- tqdm==4.65.0
- peft==0.4.0
- transformers==4.31.0
- deepspeed==0.13.0

To install all dependencies:
```
pip install -r requirements.txt
```

## Datasets
1. Install DeepMIMO package from pip by
```
pip install DeepMIMO
```
2. Select and download a scenario from the [[scenarios page]](https://www.deepmimo.net/scenarios/). Then extract scenario folder into `./deepmimo_generate_data/scenarios`.

3. Generate the dataset through running `./deepmimo_generate_data/gen_training_data.py`. Then put the generated dataset into `./dataset/`.

## Train your model
1. You can download the [[GPT2-large]](https://huggingface.co/openai-community/gpt2-large) from Hugging Face. Then put the model into `./gpt2-large`.
2. Tune the model. We provide an experiment script for training the model.
```bash
bash ./scripts/LLM_BP.sh 
```
Arguments Explanation:
- `checkpoints` : str type, indicating the file path where the checkpoint is saved.
- `speeds` : int type, indicating the speed values for the dataset used during training.
- `num_antenna`: int type, indicating the number of antennas in the dataset for training.
## Test your model
We provide an experiment script for testing the model.
```bash
bash ./scripts/LLM_BP_test.sh 
```
Arguments Explanation:
- `root_path` : str type, indicating the file path where the dataset for testing is saved.


## Acknowledgement
Our implementation adapts [OFA (GPT4TS)](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All) as the code base and have extensively modified it to our purposes. We thank the authors for sharing their implementations and related resources.

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

```
@article{sheng2024beam,
  title={Beam prediction based on large language models},
  author={Sheng, Yucheng and Huang, Kai and Liang, Le and Liu, Peng and Jin, Shi and Li, Geoffrey Ye},
  journal={arXiv preprint arXiv:2408.08707},
  year={2024}
}
```