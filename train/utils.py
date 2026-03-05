# 模組說明：提供訓練過程的視覺化、監控與記錄相關工具。
"""Visualization, monitoring and logging utils."""

# 匯入 matplotlib 的 pyplot 介面，方便建立圖表。
import matplotlib.pyplot as plt


# 定義函式：輸入 keras 訓練歷史，繪製 loss 與指定 metrics 的變化圖。
def plot_history(history, mets=['auc_10'], figsize=(15, 3)):
  # 函式說明：繪製 keras.fit 回傳的 history 曲線圖。
  """Plot keras.fit history graphs."""

  # 從 history 中找出所有「訓練 loss」鍵值（包含 loss、但不包含 val）。
  loss_list = [
      # 逐一檢查 history 的鍵值，篩選訓練 loss 名稱。
      s for s in history.history.keys() if 'loss' in s and 'val' not in s
  ]
  # 從 history 中找出所有「驗證 loss」鍵值（包含 loss 且包含 val）。
  val_loss_list = [
      # 逐一檢查 history 的鍵值，篩選驗證 loss 名稱。
      s for s in history.history.keys() if 'loss' in s and 'val' in s
  ]

  # 建立 epoch 序列（從 1 開始），長度以第一個訓練 loss 的紀錄數為準。
  epochs = range(1, len(history.history[loss_list[0]]) + 1)
  # 建立整張圖，大小由 figsize 參數控制。
  plt.figure(figsize=figsize)
  # 建立第一個子圖位置，用來畫 loss 曲線。
  ax = plt.subplot(1, 1 + len(mets), 1)

  # 逐一繪製每個訓練 loss 曲線。
  for l in loss_list:
    # 在目前子圖上繪製藍色訓練 loss，並在圖例顯示最後一個數值。
    ax.plot(
        # x 軸：epochs。
        epochs,
        # y 軸：對應 loss 歷史值。
        history.history[l],
        # 線條樣式：藍色。
        'b',
        # 圖例文字：Train loss + 最後一個 epoch 的 loss 值（格式化到小數三位）。
        label='Train loss ('
        + str(str(format(history.history[l][-1], '.3f')) + ')'),
    )
  # 逐一繪製每個驗證 loss 曲線。
  for l in val_loss_list:
    # 在目前子圖上繪製綠色驗證 loss，並在圖例顯示最後一個數值。
    ax.plot(
        # x 軸：epochs。
        epochs,
        # y 軸：對應驗證 loss 歷史值。
        history.history[l],
        # 線條樣式：綠色。
        'g',
        # 圖例文字：Valid loss + 最後一個 epoch 的 loss 值（格式化到小數三位）。
        label='Valid loss ('
        + str(str(format(history.history[l][-1], '.3f')) + ')'),
    )

  # 設定 loss 子圖標題。
  ax.set_title('Loss')
  # 設定 x 軸標籤。
  ax.set_xlabel('Epochs')
  # 設定 y 軸標籤。
  ax.set_ylabel('Loss')
  # 顯示圖例。
  ax.legend()

  # 逐一處理使用者指定要畫出的 metrics（例如 auc、accuracy、lr）。
  for i, met_name in enumerate(mets):
    # 建立對應 metric 的子圖，位置從第 2 格開始。
    ax = plt.subplot(1, 1 + len(mets), 2 + i)

    # 若 metric 為學習率 lr，直接畫出 lr 曲線（通常沒有 val 對應）。
    if met_name == 'lr':
      # 繪製藍色學習率曲線。
      ax.plot(epochs, history.history['lr'], 'b')

    # 其他 metric（例如 auc、accuracy）則同時畫訓練與驗證曲線。
    else:
      # 找到對應的訓練 metric 名稱（包含 met_name、且不含 val）。
      train_met_name = [
          # 從所有鍵值中篩選訓練 metric。
          s for s in history.history.keys() if met_name in s and 'val' not in s
      ][0]
      # 找到對應的驗證 metric 名稱（包含 met_name、且包含 val）。
      val_met_name = [
          # 從所有鍵值中篩選驗證 metric。
          s for s in history.history.keys() if met_name in s and 'val' in s
      ][0]

      # 繪製訓練 metric 曲線。
      ax.plot(
          # x 軸：epochs。
          epochs,
          # y 軸：訓練 metric 歷史值。
          history.history[train_met_name],
          # 線條樣式：藍色。
          'b',
          # 圖例文字：Train + metric 名稱 + 最後值（小數三位）。
          label=f'Train {met_name} ('
          + str(str(format(history.history[train_met_name][-1], '.3f')) + ')'),
      )
      # 繪製驗證 metric 曲線。
      ax.plot(
          # x 軸：epochs。
          epochs,
          # y 軸：驗證 metric 歷史值。
          history.history[val_met_name],
          # 線條樣式：綠色。
          'g',
          # 圖例文字：Valid + metric 名稱 + 最後值（小數三位）。
          label=f'Valid {met_name} ('
          + str(str(format(history.history[val_met_name][-1], '.3f')) + ')'),
      )

    # 設定 metric 子圖標題（使用 metric 名稱）。
    ax.set_title(met_name)
    # 設定 x 軸標籤。
    ax.set_xlabel('Epochs')
    # 設定 y 軸標籤（使用 metric 名稱）。
    ax.set_ylabel(met_name)
    # 顯示圖例。
    ax.legend()
