from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import json, time, argparse

# 帶入自訂的執行參數
parser = argparse.ArgumentParser(description='訓練情緒分類語言模型')
parser.add_argument('--data', help='訓練資料')
args = parser.parse_args()


# 讀取訓練資料
def getDataFrame():
    # 讀取 JSON 文字結構
    with open(args.data, 'r', encoding="utf8") as file:  
        listJson = json.loads(file.read()) # 將 JSON 轉成陣列

    # 將訓練資料轉成 panda DataFrame，並提供 headers
    df = pd.DataFrame(listJson)
    df.columns = ["text", "labels"]
    
    # 將 labels 欄位的資料，轉成數值，才能完整符合訓練格式
    df["labels"] = pd.to_numeric(df["labels"])

    # 回傳 DataFrame
    return df

# 訓練模型
def train(df):
    # 輸出語言模型的目錄名稱
    dir_name = 'bert-base-chinese-bs-64-epo-3' 

    # 自訂參數
    model_args = ClassificationArgs()
    model_args.train_batch_size = 64
    model_args.num_train_epochs = 3
    model_args.output_dir = f"outputs/{dir_name}"

    # 建立 ClassificationModel (會自動下載預訓練模型)
    '''
    如果僅有 CPU，可以將以下參數調整成 use_cude=False
    '''
    model = ClassificationModel(
        'bert', # 選擇 bert (simple transformers 模型代碼)
        'bert-base-chinese', # 支援中文的 bert 預訓練模型
        use_cuda=True, # 啟用 GPU
        cuda_device=0,
        num_labels=6, # multi-class 有 6 類，所以寫 6
        args=model_args # 帶入自訂參數
    )

    # 訓練 model，會將 
    model.train_model(df)


# 主程式
if __name__ == "__main__":
    tStart = time.time() # 計時開始
    train( getDataFrame() )
    tEnd = time.time() # 計時結束

    # 輸出程式執行的時間
    print(f"執行花費 {tEnd - tStart} 秒。")
