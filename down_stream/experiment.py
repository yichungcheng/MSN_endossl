"""A compact module for running down-stream experiments."""  # 模組說明：用於執行下游任務實驗。

import sys  # 匯入 sys，以便調整 Python 模組搜尋路徑。
import os  # 匯入 os，供檔案路徑與資料夾操作使用。
sys.path.append(os.path.realpath(__file__ + '/../../'))  # 將專案根目錄加入路徑，讓跨資料夾模組可被匯入。

import tensorflow as tf  # 匯入 TensorFlow，建立與訓練模型。

from data import cholec80_images  # 匯入資料集工具，建立 Cholec80 影像資料集。
from train import eval_lib  # 匯入評估函式庫，做訓練後驗證/測試評估。
from train import train_lib  # 匯入訓練函式庫，提供模型、優化器、callbacks 等工具。


def verbose_print(msg, verbose=True, is_title=False):  # 定義可控輸出的列印函式。
  if verbose:  # 若 verbose 為 True，才印出一般訊息。
    print(msg)  # 輸出傳入的訊息內容。
  if is_title:  # 若此訊息被標記為標題。
      print('#' * 40)  # 額外印出分隔線，方便閱讀日誌區段。


def run_experiment(config, verbose=True):  # 定義主流程：根據設定執行完整實驗。
  """Stand-alone function for running an experiment."""  # 函式說明：可獨立呼叫以執行一次實驗。

  verbose_print('Config:', verbose)  # 先印出「Config」標題。
  attr_names = [i for i in dir(config) if not i.startswith('__')]  # 取得 config 物件中可見且非內建屬性名稱。
  for a in attr_names:  # 逐一走訪每個設定欄位。
    verbose_print('{} = {}'.format(a, getattr(config, a, None)), verbose)  # 印出欄位名稱與對應值。
  print('\n\n')  # 輸出空行，分隔設定區與後續流程。

  if not os.path.exists(config.exp_dir):  # 若實驗輸出目錄不存在。
    os.makedirs(config.exp_dir)  # 建立實驗輸出目錄。

  ##############################################################################  # 區塊分隔：資料集與模型建立。
  # Datasets & Model  # 區塊標題：資料集與模型。
  ##############################################################################  # 區塊分隔線。
  verbose_print('Create datasets and model', verbose, True)  # 印出建立資料與模型的標題訊息。
  datasets = cholec80_images.get_cholec80_images_datasets(  # 呼叫資料函式，取得 train/validation/test 資料集。
      data_root=config.data_root,  # 指定資料根目錄。
      batch_size=config.batch_size,  # 指定批次大小。
      train_transformation=config.train_transformation,  # 指定訓練資料增強/轉換設定。
  )  # 結束資料集建立呼叫。


  if config.is_linear_evaluation:  # 若設定為線性評估模式（凍結 backbone，只訓練線性層）。
    model = train_lib.get_linear_model(  # 建立線性分類模型。
        input_dim=config.input_dim, output_dim=config.num_classes  # 設定輸入特徵維度與輸出類別數。
    )  # 結束線性模型建立。
  elif config.model == 'resnet50':  # 否則若模型類型指定為 ResNet50V2。
    input_tensor = tf.keras.Input(shape=(224, 224, 3,))  # 建立模型輸入張量（224x224 RGB）。
    backbone = tf.keras.applications.resnet_v2.ResNet50V2(  # 建立 ResNet50V2 主幹網路。
        include_top=False,  # 不使用 ImageNet 原本頂層分類器。
        weights='imagenet',  # 載入 ImageNet 預訓練權重。
        input_tensor=input_tensor)  # 將自訂輸入張量接入 backbone。
    out = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)  # 對特徵圖做全域平均池化。
    out = tf.keras.layers.Dense(config.num_classes)(out)  # 接上分類 Dense 層，輸出類別 logits。
    model = tf.keras.Model(input_tensor, out, name='Model')  # 將輸入與輸出封裝為完整 Keras 模型。
  elif 'vit' in config.model:  # 否則若模型名稱含 vit（Vision Transformer）。
    backbone = tf.saved_model.load(config.saved_model_dir)  # 從 saved_model_dir 載入預訓練 ViT backbone。
    model = train_lib.LinearFineTuneModel(backbone, config.num_classes)  # 包裝為可微調的線性分類模型。
  else:  # 若上述模型條件皆不符合。
    raise ValueError('Invalid model name: {}'.format(config.model))  # 拋出錯誤，提示無效模型名稱。

  model.compile(  # 編譯模型，設定優化器、損失函數、評估指標。
      optimizer=train_lib.get_optimizer(  # 依 config 建立優化器。
          config.optimize_name,  # 優化器名稱（如 SGD/Adam）。
          config.learning_rate,  # 學習率。
          config.momentum,  # 動量參數（若優化器支援）。
          config.weight_decay,  # 權重衰減係數。
      ),  # 結束優化器設定。
      loss=train_lib.get_loss(config.task_type),  # 依任務型態取得對應 loss。
      metrics=train_lib.get_metrics(config.task_type, config.num_classes),  # 依任務型態與類別數設定 metrics。
  )  # 結束 compile。

  ##############################################################################  # 區塊分隔：開始訓練。
  # Train  # 區塊標題：訓練。
  ##############################################################################  # 區塊分隔線。
  verbose_print('Begin training', verbose, True)  # 印出訓練開始標題。
  history = model.fit(  # 執行模型訓練，並回傳歷史紀錄。
      datasets['train'],  # 使用訓練資料集。
      batch_size=config.batch_size,  # 訓練批次大小。
      epochs=config.num_epochs,  # 訓練 epoch 數。
      validation_data=datasets['validation'],  # 設定驗證資料集。
      class_weight=(cholec80_images._CHOLEC80_PHASES_WEIGHTS  # 若啟用，使用類別權重平衡不平衡資料。
                    if config.use_class_weight else None),  # 否則不使用 class weight。
      callbacks=train_lib.get_callbacks(  # 建立並設定訓練 callbacks。
          callbacks_names=config.callbacks_names,  # callback 名稱清單。
          exp_dir=config.exp_dir,  # 實驗輸出路徑（存 checkpoint/log）。
          monitor_metric=config.monitor_metric,  # callback 監控指標。
          learning_rate=config.learning_rate,  # callback 可能需要的初始學習率。
      ),  # 結束 callbacks 設定。
      validation_freq=config.validation_freq,  # 每幾個 epoch 做一次驗證。
  )  # 結束 fit 呼叫。

  #############################################################################  # 區塊分隔：訓練結束後評估。
  # End of train evaluation  # 區塊標題：訓練結束評估。
  #############################################################################  # 區塊分隔線。
  if config.manually_load_best_checkpoint:  # 若設定需要手動載入最佳/最新 checkpoint。
    checkpoints = os.listdir(os.path.join(config.exp_dir, 'checkpoints') + '/*')  # 嘗試讀取 checkpoint 路徑下的檔案清單。
    if checkpoints:  # 若有找到 checkpoint。
      latest = checkpoints[-1]  # 取清單最後一個作為最新 checkpoint。
      print(f'Load latest checkpoint: {latest}')  # 印出即將載入的 checkpoint 路徑。
      model.load_weights(latest)  # 載入 checkpoint 權重。
    else:  # 若沒有 checkpoint 可載入。
      print('Haven\'t loaded a saved checkpoint')  # 提示未載入任何已儲存權重。

  # For the special case of phases, re-extract the dataset with the 'with_image_path'  # 說明：phase 任務需重新取資料，附帶影像路徑。
  # attribute for calculating video-level metrics  # 原因：後續要計算影片層級指標。
  verbose_print('Start end of train evaluation', verbose, True)  # 印出進入訓練後評估流程的標題。
  datasets = cholec80_images.get_cholec80_images_datasets(  # 重新建立資料集（含 image path）。
      data_root=config.data_root,  # 指定資料根目錄。
      batch_size=config.batch_size,  # 指定批次大小。
      train_transformation=config.train_transformation,  # 保持相同訓練轉換設定。
      with_image_path=True,  # 額外回傳影像路徑，供影片層級評估使用。
  )  # 結束資料集重建。

  mets = eval_lib.end_of_training_evaluation(  # 執行訓練結束評估並取得指標。
      model,  # 傳入已訓練模型。
      datasets['validation'],  # 傳入驗證集。
      datasets['test'],  # 傳入測試集。
      label_key=config.label_key,  # 指定標籤欄位名稱。
      exp_dir=config.exp_dir,  # 指定輸出目錄（儲存評估結果）。
      epoch=config.num_epochs)  # 指定評估所屬 epoch（通常為最終 epoch）。
  return mets, history  # 回傳評估指標與訓練歷史。
