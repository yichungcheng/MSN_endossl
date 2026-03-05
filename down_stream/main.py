"""主程式：用於線性評估（linear evaluation）或微調（fine-tuning）。"""

# 匯入 argparse，用來解析命令列參數。
import argparse
# 匯入本專案的設定模組，提供 Config 類別。
import config
# 匯入實驗執行模組，負責啟動訓練/評估流程。
import experiment


# 建立命令列參數解析器物件。
parser = argparse.ArgumentParser()

# 以下皆為可選參數，主要用於超參數掃描（parameter sweep）。
# 新增 --model 參數，讓使用者指定模型名稱。
parser.add_argument(
  '--model',
  type=str,
  required=False)

# 新增 --optimizer 參數，讓使用者指定最佳化器類型。
parser.add_argument(
  '--optimizer',
  type=str,
  required=False)

# 新增 --learning_rate 參數，讓使用者覆寫學習率。
parser.add_argument(
  '--learning_rate',
  type=float,
  required=False)

# 新增 --weight_decay 參數，讓使用者覆寫權重衰減值。
parser.add_argument(
  '--weight_decay',
  type=float,
  required=False)


# 定義主函式，接收解析後的命令列參數。
def main(args):
    # 建立預設設定物件（包含專案預設超參數）。
    conf = config.Config()
    # 逐一巡覽命令列參數（名稱 k、值 v）。
    for k, v in vars(args).items():
        # 只有當參數有提供有效值時才覆寫設定。
        if v:
            # 將命令列提供的值寫回設定物件對應欄位。
            setattr(conf, k, v)
    # 依照最終設定執行實驗流程（回傳值此處不使用）。
    _ = experiment.run_experiment(conf)

# 確保只有直接執行此檔案時才會啟動主程式。
if __name__ == '__main__':
  # 解析命令列輸入為 args 物件。
  args = parser.parse_args()
  # 呼叫主函式啟動流程。
  main(args)
