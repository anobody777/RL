# ==========================================
# 0. 安裝必要的套件 (包含 yfinance)
# ==========================================


import twstock
import yfinance as yf  # 引入 Yahoo Finance
import pandas as pd
import numpy as np
import time
import random
from prettytable import PrettyTable
from tqdm import tqdm

# ==========================================
# 1. 設定參數與清單
# ==========================================
# 台灣50成分股 (範例)
tw50_tickers = [
    '2330', '2317', '2454', '2382', '2308', '2881', '2303', '2882', '2891', '3711',
    '2886', '2884', '2892', '1216', '5880', '2357', '2885', '5871', '2327', '2412',
    '2002', '3231', '3034', '2880', '3037', '2603', '2890', '2883', '2887', '3008',
    '2395', '2352', '1101', '3045', '5876', '1301', '1303', '4904', '2912', '4938',
    '2379', '1326', '6505', '1402', '2301', '6669', '2615', '2609', '9910', '6415'
]

# 參數設定
LOOKBACK_DAYS = 3      # 累積天數
MA_PERIOD = 20         # 均線週期
BIAS_LIMIT = 10        # 乖離率限制
Z_SCORE_THRESHOLD = 1.0 # Z-Score 門檻

results = []

print(f"🚀 啟動「混合雙引擎」篩選模型 (Yahoo Finance + twstock)...")
print("-" * 60)

# ==========================================
# 2. 執行掃描
# ==========================================
for code in tqdm(tw50_tickers, desc="分析進度"):
    try:
        # --- 引擎 A: Yahoo Finance (負責抓股價/均線) ---
        # Yahoo 的台股代碼需要加上 ".TW"
        yf_ticker = f"{code}.TW"
        stock_yf = yf.Ticker(yf_ticker)
        
        # 抓取歷史資料 (period='3mo' 抓三個月，確保均線計算無誤)
        df_price = stock_yf.history(period="3mo")
        
        if len(df_price) < MA_PERIOD:
            continue # 資料不足

        # 計算技術指標
        current_price = df_price['Close'].iloc[-1]
        ma_20 = df_price['Close'].rolling(window=MA_PERIOD).mean().iloc[-1]
        bias = ((current_price - ma_20) / ma_20) * 100
        
        # 取得當日成交量 (Yahoo 的 Volume 單位是「股」或「筆」，通常需要檢查)
        # 這裡直接用 Volume，Yahoo 通常給的是股數
        current_volume_sheets = df_price['Volume'].iloc[-1] / 1000 # 換算成「張」

        # [第一層濾網] 技術面
        if (current_price < ma_20) or (bias > BIAS_LIMIT):
            continue

        # --- 引擎 B: twstock (負責抓三大法人籌碼) ---
        stock_tw = twstock.Stock(code)
        
        # 這裡只呼叫 fetch_3i (抓法人)，不要呼叫 stock_tw.fetch() (抓股價會報錯)
        history_3i = stock_tw.fetch_3i(days=60)
        
        if not history_3i or len(history_3i) < 30:
            continue

        # 整理籌碼數據
        foreign_list = [v['foreign']['total'] for v in history_3i]
        trust_list = [v['trust']['total'] for v in history_3i]
        dealer_list = [v['dealer']['total'] for v in history_3i]
        
        df_chips = pd.DataFrame({
            'Foreign': foreign_list,
            'Trust': trust_list,
            'Dealer': dealer_list
        })
        df_chips = df_chips / 1000 # 換算張數
        df_chips['Smart_Money'] = df_chips['Foreign'] + df_chips['Trust']

        # [第二層濾網] 籌碼累積
        # 為了避免 Yahoo 和 twstock 日期對不齊，我們直接取籌碼資料的「最後 N 筆」
        recent_smart_sum = df_chips['Smart_Money'].tail(LOOKBACK_DAYS).sum()
        recent_dealer_sum = df_chips['Dealer'].tail(LOOKBACK_DAYS).sum()

        if (recent_smart_sum <= 0) or (recent_dealer_sum > 0):
            continue

        # [第三層濾網] Z-Score 統計
        rolling_smart = df_chips['Smart_Money'].rolling(window=LOOKBACK_DAYS).sum()
        baseline = rolling_smart.tail(20 + LOOKBACK_DAYS)
        mean_val = baseline.mean()
        std_val = baseline.std()

        if std_val == 0:
            z_score = 0
        else:
            z_score = (recent_smart_sum - mean_val) / std_val

        if z_score > Z_SCORE_THRESHOLD:
            results.append({
                'Stock': code,
                'Price': round(current_price, 2),
                'Bias(%)': round(bias, 2),
                'Vol(張)': int(current_volume_sheets),
                '主力3日買': int(recent_smart_sum),
                'Z-Score': round(z_score, 2)
            })

    except Exception as e:
        # 遇到錯誤跳過，但不中斷
        # print(f"Error {code}: {e}") 
        pass
    
    # 流量控制 (Yahoo 比較耐操，但 twstock 還是要休息)
    time.sleep(random.uniform(2, 4))

# ==========================================
# 3. 輸出報表
# ==========================================
if results:
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by='Z-Score', ascending=False).reset_index(drop=True)
    
    table = PrettyTable()
    table.field_names = df_res.columns.tolist()
    for row in df_res.itertuples(index=False):
        table.add_row(row)
        
    print(f"\n📊 【混合引擎】篩選結果 (共 {len(df_res)} 檔)：")
    print(table)
else:
    print("\n⚠️ 執行完成，但沒有股票符合條件。")
    print("建議：嘗試將 Z_SCORE_THRESHOLD 調降至 0.5 或 0 測試系統運作。")