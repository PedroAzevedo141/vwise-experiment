# VWise Experiment



## Getting started

Crie um ambiente virtual e instale as dependências:

```bash
conda create -n vwise python=3.8 --yes
conda activate vwise
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
```

## Dataset

- [ ] [Baixe](https://cinufpe.sharepoint.com/:u:/s/VOXAR-BACKUP/EVEaVSqeFipOhJASRZtSZAEBXNMNc4nfwDlPIvGlTN5AJw?e=yemT4U) o dataset e extraia os arquivos na pasta `data/` na raiz do projeto.

## Modelos pré-treinados

- [ ] [Baixe o EfficientNet](https://drive.google.com/file/d/17_V4hGYEFuiJq2P8MF0EiNTyP3u4Cz8F/view?usp=sharing) e coloque o modelo na pasta `efficientnet_v2/pretrained/`.
- [ ] [Baixe o MobileNet_V3](https://drive.google.com/file/d/17dPAdcNxJ84khLY9ZMOSK9OApc6UCdM3/view?usp=sharing) e coloque o modelo na pasta `mobilenet_v3/pretrained/`.
- [ ] [Baixe o VisionTransformers](https://drive.google.com/file/d/17vmNjIlVWjYXU9NJjgK6kDwyv4ZAs9oi/view?usp=sharing) e coloque o modelo na pasta `VisionTransfomers/pretrained/`.

## Quickstart

```bash
cd efficientnet_v2
python demo.py
```