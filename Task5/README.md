# Lab 5: Multi-Modal Sentiment Analysis

This is the repository of Modern AI course's final project in ECNU.

## Environment

This implemetation is based on Python3. To run the code, you need the following dependencies:

- pandas==2.2.3

- scikit_learn==1.6.1

- torch==2.5.1

- torchvision==0.20.1

- transformers==4.47.1

You can simply run 

```python
pip install -r requirements.txt
```

## Repository Structure

I select some important files for detailed description.

```python
|-- aggmodel.py  # basic model and ablation test model code
|-- data.py  # code of creating dataloaders
|-- main.py  # the main code
|-- dataset
    |-- data  # original dataset
    |-- data_prepare.py  # split the orginal dataset into train & val set
    |-- test  # test set
    |-- train  # train set
    `-- val  # val set
    |-- test_without_label.txt  # test set prediction results
    |-- train.txt
```

## Run This Code

* To split the data to train set and val set,use:

```powershell
python data_prepare.py
```

* To run the full model, use:

```powershell
python main.py --model agg --lr 1e-5
```

* To run the ablation test, use:

```powershell
python main.py --model text --lr 1e-5
python main.py --model image --lr 1e-5
```

## Attribution

Part of this code are based on the following repositories:

[BERT](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)

[ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)

