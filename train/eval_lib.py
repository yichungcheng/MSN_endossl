"""Evaluation methods."""  # 模組說明：集中放置模型評估相關函式。

import numpy as np  # 匯入 NumPy，用於陣列操作與統計計算。
from sklearn.metrics import average_precision_score  # 匯入 AP（Average Precision）計算函式。
from sklearn.metrics import classification_report  # 匯入分類報告函式（可回傳 accuracy、precision、recall、f1）。
from sklearn.metrics import precision_recall_fscore_support as score  # 匯入 precision/recall/f1/support 計算函式並命名為 score。
import tensorflow as tf  # 匯入 TensorFlow，用於模型推論與 TensorBoard 紀錄。


def calc_f1(model, ds, agg='video', verbose=0):  # 定義 F1 計算函式；可選擇 frame 或 video 層級聚合。
  video2labels = {}  # 建立字典：記錄每個影片對應的真實標籤序列。
  video2preds = {}  # 建立字典：記錄每個影片對應的預測標籤序列。

  all_labels = []  # 建立清單：累積所有 frame/clip 的真實標籤。
  all_preds = []  # 建立清單：累積所有 frame/clip 的預測標籤。
  for batch in ds:  # 逐批走訪資料集。

    inputs, labels, clip_paths = batch  # 拆出輸入、標籤與 clip 路徑。
    preds = model.predict(inputs, verbose=verbose)  # 以模型對該批輸入做推論。
    preds = tf.argmax(preds, 1)  # 將類別機率/分數轉成最終預測類別索引。

    all_labels += labels.numpy().tolist()  # 將該批真實標籤轉成 Python list 後併入總清單。
    all_preds += preds.numpy().tolist()  # 將該批預測標籤轉成 Python list 後併入總清單。

    if agg == 'video':  # 若要以影片層級評估，需按影片 regroup 各 clip 結果。
      for c, l, p in zip(clip_paths.numpy(), labels.numpy(), preds.numpy()):  # 同步迭代每個 clip 的路徑、標籤與預測。
        c = c.decode('utf-8').split('/')[-2]  # 從路徑字串擷取影片 ID（倒數第二層資料夾名）。
        if c not in video2labels:  # 若該影片尚未建立儲存空間。
          video2labels[c] = []  # 初始化該影片的真實標籤清單。
          video2preds[c] = []  # 初始化該影片的預測標籤清單。
        video2labels[c].append(l)  # 將目前 clip 的真實標籤加入對應影片。
        video2preds[c].append(p)  # 將目前 clip 的預測標籤加入對應影片。

  # anyway calculate frame-level metrics  # 註解：無論最終聚合方式，都先計算 frame 層級指標供後續使用。
  all_labels = np.asarray(all_labels)  # 將總真實標籤清單轉為 NumPy 陣列。
  all_preds = np.asarray(all_preds)  # 將總預測標籤清單轉為 NumPy 陣列。
  frame_mets = classification_report(all_labels, all_preds, output_dict=True)  # 產生 frame 層級分類報告（字典格式）。

  if agg == 'frame':  # 若指定回傳 frame 層級結果。
    all_labels = np.asarray(all_labels)  # 確保真實標籤為 NumPy 陣列。
    all_preds = np.asarray(all_preds)  # 確保預測標籤為 NumPy 陣列。

    return {  # 回傳 frame 層級主要評估指標。
        'acc': np.round(  # 計算並四捨五入 accuracy（百分比）。
            np.sum(all_labels == all_preds) * 100 / len(all_labels), 2  # 以正確數/總數計算準確率。
        ),  # 結束 acc 計算。
        'f1': np.round(score(all_labels, all_preds)[-2].mean() * 100, 2),  # 取各類別 f1 平均後轉成百分比並四捨五入。
    }  # 結束 frame 模式回傳。

  elif agg == 'video':  # 若指定回傳 video 層級結果。
    accs = []  # 建立清單：存每部影片的準確率。
    scores = []  # 建立清單：存每部影片的 precision/recall/f1/support 平均值。
    for sub_labels, sub_preds in zip(  # 逐部影片配對真實標籤序列與預測序列。
        video2labels.values(), video2preds.values()  # 取出所有影片的標籤與預測集合。
    ):  # 結束影片逐一迭代定義。
      sub_labels = np.asarray(sub_labels)  # 該影片真實標籤轉為 NumPy 陣列。
      sub_preds = np.asarray(sub_preds)  # 該影片預測標籤轉為 NumPy 陣列。

      # compute acc and append  # 註解：先計算單一影片準確率並加入清單。
      vid_acc = np.sum(sub_labels == sub_preds) * 100 / len(sub_labels)  # 計算該影片準確率（百分比）。
      accs.append(vid_acc)  # 將該影片準確率加入總清單。

      # compute F1  # 註解：計算單一影片 precision/recall/f1 等指標。
      vid_score = score(sub_labels, sub_preds)  # 取得該影片各類別 precision/recall/f1/support。
      mean = np.mean(np.vstack(vid_score).T, axis=0)  # 對各類別做平均，得到單一影片整體指標。
      mean[:-1] *= 100  # 將 precision/recall/f1 轉為百分比（support 不轉）。
      scores.append(mean)  # 將該影片指標加入總清單。

    # summarize  # 註解：彙整所有影片的指標。
    overall_acc = np.around(np.mean(np.stack(accs)), 2)  # 計算所有影片平均準確率並四捨五入。
    overall_f1 = np.mean(np.stack(scores), axis=0)  # 計算所有影片平均 precision/recall/f1/support。
    overall_f1 = np.around(overall_f1, 2)  # 對上述平均結果做四捨五入。

    return {  # 回傳 video 與 frame 層級摘要結果。
        'video_acc': overall_acc,  # 影片層級平均準確率。
        'video_precision': overall_f1[0],  # 影片層級平均 precision。
        'video_recall': overall_f1[1],  # 影片層級平均 recall。
        'video_f1': overall_f1[2],  # 影片層級平均 f1。

        'frame_acc': np.around(  # frame 層級 accuracy（百分比）。
            frame_mets['accuracy'] * 100, 2),  # 從分類報告取 accuracy 後轉百分比並四捨五入。
        'frame_macro_f1': np.around(  # frame 層級 macro-f1。
            frame_mets['macro avg']['f1-score'] * 100, 2),  # 從分類報告取 macro avg f1-score。
        'frame_micro_f1': np.around(  # frame 層級 weighted/micro 風格 f1（原程式使用 weighted avg）。
            frame_mets['weighted avg']['f1-score'] * 100, 2),  # 從分類報告取 weighted avg f1-score。
        }  # 結束 video 模式回傳。


def calc_map(model, ds, agg='all', verbose=0):  # 定義 mAP 計算函式；可回傳整體或分類別結果。

  all_labels = np.empty(())  # 先建立空陣列作為標籤容器（稍後會被第一批資料覆蓋）。
  all_preds = np.empty(())  # 先建立空陣列作為預測容器（稍後會被第一批資料覆蓋）。
  for i, batch in enumerate(ds):  # 逐批走訪資料集並取得批次索引。
    inputs, labels = batch  # 拆出輸入與標籤（此任務不含 clip 路徑）。
    preds = model.predict(inputs, verbose=verbose)  # 對該批輸入做推論。
    if i == 0:  # 若為第一批，直接初始化總容器。
      all_labels = labels.numpy()  # 以第一批真實標籤初始化。
      all_preds = preds  # 以第一批預測結果初始化。
    else:  # 若非第一批，則與既有資料串接。
      all_labels = np.concatenate((all_labels, labels.numpy()), 0)  # 在第 0 維串接真實標籤。
      all_preds = np.concatenate((all_preds, preds), 0)  # 在第 0 維串接預測分數。
  try:  # 嘗試計算 mAP，避免評估時因資料異常中斷。
    mean = [mean_ap(all_labels, all_preds) * 100]  # 計算整體 mAP 並轉百分比（包成 list 方便統一格式）。
    std = [0.00]  # 此實作未估標準差，固定回傳 0。
    if agg == 'class':  # 若要求分類別結果。
      mean = mean_ap(all_labels, all_preds, mean=False) * 100  # 改為取得每一類別的 AP。
      std = [0.00] * 7  # 分類別模式下回傳對應長度的 0 標準差陣列（此處固定 7 類）。
  except:  # 若計算 AP 發生例外（例如標籤分布不合法）。
    mean = [-1] * 7 if agg == 'class' else [-1.00]  # 失敗時以 -1 當作指標無效值。
    std = [0.00] * 7 if agg == 'class' else [0.00]  # 失敗時標準差維持 0。
  mean = [np.round(i, 2) for i in mean]  # 將 mean 結果逐一四捨五入到小數點後兩位。
  std = [np.round(i, 2) for i in std]  # 將 std 結果逐一四捨五入到小數點後兩位。
  if len(mean) == 1:  # 若只有單一整體指標。
    return {'mean': mean[0], 'std': std[0]}  # 回傳標量格式的 mean/std。
  return {'mean': mean, 'std': std}  # 否則回傳向量格式的分類別 mean/std。


def mean_ap(labels, predictions, mean=True):  # 定義 AP 計算輔助函式。
  metrics = np.array(average_precision_score(labels, predictions, average=None))  # 計算各類別 AP 並轉為 NumPy 陣列。
  if mean:  # 若需要整體平均 AP。
    metrics = np.sum([x for x in metrics]) / len(metrics)  # 以各類 AP 的算術平均作為 mAP。
  return metrics  # 回傳單一 mAP 或各類別 AP。

def end_of_training_evaluation(  # 定義訓練結束時的整體評估函式。
    model, validation_ds, test_ds, label_key, exp_dir, epoch):  # 輸入模型、驗證/測試集、任務類型、輸出目錄與 epoch。
  """Run aggregative evaluation over the trained model."""  # 函式說明：對訓練完成模型執行聚合評估。

  if label_key == 'tool':  # 若任務標籤為 tool（多標籤 mAP 任務）。
    validation_map = calc_map(model, validation_ds)  # 計算驗證集 mAP。
    test_map = calc_map(model, test_ds)  # 計算測試集 mAP。
    mets = {'val_map': validation_map['mean'], 'test_map': test_map['mean']}  # 整理成統一字典輸出。

  elif label_key == 'segment':  # 若任務標籤為 segment（分類 F1 任務）。
    mets = {}  # 建立空字典存放各項指標。
    for k, v in calc_f1(model, validation_ds, agg='video').items():  # 計算驗證集 video 聚合指標並逐項走訪。
      mets[f'val_{k}'] = v  # 將驗證集指標加上 val_ 前綴存入結果。
    for k, v in calc_f1(model, test_ds, agg='video').items():  # 計算測試集 video 聚合指標並逐項走訪。
      mets[f'test_{k}'] = v  # 將測試集指標加上 test_ 前綴存入結果。

  else:  # 若 label_key 非預期值。
    raise ValueError  # 拋出錯誤提醒呼叫端輸入不合法。

  file_writer = tf.summary.create_file_writer(exp_dir + '/metrics')  # 建立 TensorBoard writer，輸出到 metrics 子目錄。
  file_writer.set_as_default()  # 設定此 writer 為預設 writer。
  for k, v in mets.items():  # 逐一走訪所有評估指標。
    tf.summary.scalar(k, data=v, step=epoch)  # 以 scalar 形式寫入 TensorBoard，step 使用當前 epoch。
  return mets  # 回傳最終評估指標字典。
