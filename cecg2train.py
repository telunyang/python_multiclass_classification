import sys, json, time, argparse

parser = argparse.ArgumentParser(description='建立訓練資料')
parser.add_argument('--data', help='CECG 資料集 (JSON 格式)')
parser.add_argument('--save', help='將訓練資料儲存到哪裡 (JSON 格式)')
args = parser.parse_args()

# 讀取 CECG 資料集，整合成訓練格式
def getData():
    # 放置每一組 CECG 對話的變數
    listCECG = []

    # 讀取 JSON 文字結構
    with open(args.data, 'r', encoding="utf8") as file:
        '''
        從 listJSON 取得每一組單論對話 listDoc，
        其中 listDoc 元素有兩個 list，分別是:
        => ["現在 刷 朋友 圈 最大 的 快樂 就是 看 代 購 們 各種 直播 。 。 。 。 。", "5"]
        => ["臥 槽 我 也 是", "4"]
        '''
        # 將 JSON 轉成 list
        listJson = json.loads(file.read()) 
        # 走訪每一組對話
        for listDoc in listJson: 
            # 整合所有對話
            listCECG += listDoc
    
    # 回傳符合訓練格式的對話資料
    return listCECG

# 將 CECG 語料，儲存成 [text, label] 的 JSON 格式 
def saveData(listData):
    with open(args.save, 'w', encoding='utf-8') as file:
        file.write( json.dumps(listData, ensure_ascii=False) )

# 主程式 
try: 
    tStart = time.time() #計時開始 
    saveData( getData() )
    tEnd = time.time() #計時結束
    print(f"執行花費 {tEnd - tStart} 秒。")
except: 
    print("Unexpected error: ", sys.exc_info()) 