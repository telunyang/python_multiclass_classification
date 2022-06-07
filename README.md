# 使用 Simple Transformers 建立情緒分類模型 (multiclass classification)
Transformers 是一種深度學習的神經網路模型架構，主要用於自然語言處理（NLP，Natural Language Processing），完成文本分類、資訊擷取、問答、摘要、翻譯、文本生成等任務，提供 BERT、GPT-2、RoBERTa、T5等支援 100 種以上語系的預訓練模型，讓開發者能夠直接使用，或透過微調（fine-tune）來進行客製化。
Simple Transformers 是基於 Transformers 的一種函式庫，可以讓使用者快速地訓練出自定義的語言模型。本文將使用 Simple Transformers來建立一個多元（multi-class）情緒分類模型，協助開發者完成未知語料的情緒分類任務。

## 技術文章來源
尚未刊出

## 套件安裝
- 有 GPU
  - `pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`
- 只有 CPU
  - `pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`
- 安裝 simple transformers
  - `pip install simpletransformers`

## 資料集下載 (STC-3 CECG)
[https://drive.google.com/drive/folders/1EmqZsb3Lp_M7ftSiKVgHC6xIiWQVmDBe?usp=sharing](https://drive.google.com/drive/folders/1EmqZsb3Lp_M7ftSiKVgHC6xIiWQVmDBe?usp=sharing)

## 檔案說明
- cecg2train.py
  - 將 CECG 資料集轉成 training data
  - 指令: `python cecg2train.py --data=stc3_cecg_2017_and_2019_170w.json --save=train.json`
- train.py
  - 將 training data 進行訓練，建立模型
  - 指令: `python train.py --data=train.json`
- predict.py
  - 預測文句的情緒分類
  - 指令: `python predict.py`

## 參考資料
1. 自然語言處理
[https://zh.wikipedia.org/zh-hant/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86](https://zh.wikipedia.org/zh-hant/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86)
2. Simple Transformers
[https://simpletransformers.ai/](https://simpletransformers.ai/)
3. 範例程式碼
[https://github.com/telunyang/python_multiclass_classification](https://github.com/telunyang/python_multiclass_classification)
4. Hugging Face
[https://huggingface.co/](https://huggingface.co/)
5. Multiclass, Multilabel 以及 Multitask 的區別
[https://cynthiachuang.github.io/Difference-between-Multiclass-Multilabel-and-Multitask-Problem/](https://cynthiachuang.github.io/Difference-between-Multiclass-Multilabel-and-Multitask-Problem/)
6. Epoch, Batch size, Iteration, Learning Rate
[https://medium.com/%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7-%E5%80%92%E5%BA%95%E6%9C%89%E5%A4%9A%E6%99%BA%E6%85%A7/epoch-batch-size-iteration-learning-rate-b62bf6334c49](https://medium.com/%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7-%E5%80%92%E5%BA%95%E6%9C%89%E5%A4%9A%E6%99%BA%E6%85%A7/epoch-batch-size-iteration-learning-rate-b62bf6334c49)
7. transformers
[https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
8. NTCIR-14 Short Text Conversation Task (STC-3)
[http://sakailab.com/ntcir14stc3/](http://sakailab.com/ntcir14stc3/)
