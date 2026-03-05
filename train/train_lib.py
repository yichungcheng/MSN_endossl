"""Helper training functions."""  # 模組說明：提供模型訓練時會重複使用的工具函式。

import functools  # 匯入 functools（目前此檔案未使用，可能預留給後續函式包裝用途）。
import tensorflow as tf  # 匯入 TensorFlow 主套件，供模型、損失函式與 callback 使用。
from tensorflow_addons import metrics as tfa_metrics  # 匯入 TensorFlow Addons 指標模組並命名為 tfa_metrics。
from tensorflow_addons import optimizers as tfa_optimizers  # 匯入 TensorFlow Addons 最佳化器模組並命名為 tfa_optimizers。


def get_linear_model(input_dim: int, output_dim: int):  # 定義函式：建立簡單線性模型。
  return tf.keras.Sequential([  # 回傳 Keras Sequential 模型容器。
      tf.keras.layers.Flatten(input_shape=(input_dim,)),  # 第一層：將輸入攤平成一維向量。
      tf.keras.layers.Dense(output_dim),  # 第二層：全連接層，輸出指定維度 logits。
  ])  # 結束模型層列表。


class LinearFineTuneModel(tf.keras.Model):  # 定義自訂模型類別：以 backbone 特徵再接一層投影。
  def __init__(self, backbone, output_dim):  # 建構子：接收主幹模型與輸出維度。
    super().__init__()  # 初始化父類別 tf.keras.Model。
    self.backbone = backbone  # 儲存 backbone 模型供前向傳播使用。
    self.projection = tf.keras.layers.Dense(output_dim)  # 建立輸出投影層（分類頭）。

  def call(self, x):  # 定義前向傳播邏輯。
    return self.projection(self.backbone(x)[1])  # 取 backbone 回傳結果的第 2 個元素並投影成最終輸出。

  def get_config(self):  # 提供模型序列化時的設定取得介面。
    return super().get_config()  # 直接回傳父類別預設設定。


def get_loss(task_type: str):  # 定義函式：依任務型態選擇損失函式。
  if task_type == 'multi_class':  # 若為多類別單標籤分類。
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # 使用稀疏類別交叉熵（輸入為 logits）。
  elif task_type == 'multi_label':  # 若為多標籤分類。
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)  # 使用二元交叉熵（輸入為 logits）。

  else:  # 若任務型態不在支援清單。
    raise ValueError  # 拋出錯誤提醒呼叫端提供正確 task_type。


def get_optimizer(  # 定義函式：依名稱建立最佳化器。
    optimizer_name: str,  # 參數：最佳化器名稱。
    learning_rate: float = 0.001,  # 參數：學習率預設值。
    momentum: float = 0.9,  # 參數：動量預設值。
    weight_decay: float = 0.00001) -> tf.keras.optimizers.Optimizer:  # 參數：權重衰減，並標註回傳型別。
  """Initialize optimizer by its name."""  # 函式說明：透過名稱初始化最佳化器。

  optimizer_name = optimizer_name.lower()  # 將名稱轉小寫，避免大小寫差異造成判斷錯誤。
  if optimizer_name == 'adam':  # 分支：Adam。
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)  # 建立 Adam 並設定學習率。
  elif optimizer_name == 'adamw':  # 分支：AdamW。
    return tfa_optimizers.AdamW(  # 建立 Addons 版本 AdamW。
        learning_rate=learning_rate, weight_decay=weight_decay)  # 設定學習率與權重衰減。
  elif optimizer_name == 'rmsprop':  # 分支：RMSprop。
    return tf.keras.optimizers.RMSprop(  # 建立 RMSprop。
        learning_rate=learning_rate, momentum=momentum)  # 設定學習率與動量。
  elif optimizer_name == 'momentum':  # 分支：帶動量的 SGD。
    return tf.keras.optimizers.SGD(  # 建立 SGD（Momentum 形式）。
        learning_rate=learning_rate,  # 設定學習率。
        momentum=momentum,  # 設定動量。
        weight_decay=weight_decay)  # 設定權重衰減。
  elif optimizer_name == 'sgd':  # 分支：純 SGD。
    return tf.keras.optimizers.SGD(  # 建立 SGD。
        learning_rate=learning_rate, momentum=0, weight_decay=weight_decay)  # 動量固定為 0，並設定其他參數。
  else:  # 若名稱不支援。
    raise ValueError('Optimizer %s not supported' % optimizer_name)  # 拋出包含名稱的錯誤訊息。


class MyF1Score(tfa_metrics.F1Score):  # 自訂 F1 指標：包裝 Addons F1Score 以適配標籤格式。
  def update_state(self, y_true, y_pred, sample_weight=None):  # 覆寫狀態更新方法。
    y_true = tf.squeeze(tf.one_hot(y_true, self.num_classes), 1)  # 將稀疏標籤轉 one-hot，並移除多餘維度。
    super().update_state(y_true, y_pred, sample_weight)  # 呼叫父類別邏輯完成指標累積。


def get_metrics(task_type: str, num_classes: int):  # 定義函式：依任務型態回傳評估指標清單。
  if task_type == 'multi_class':  # 多類別任務分支。
    return [MyF1Score(num_classes=num_classes,  # 微平均 F1。
                      average='micro',  # 設定平均方式為 micro。
                      name='micro_f1'),  # 指標名稱。
            MyF1Score(num_classes=num_classes,  # 宏平均 F1。
                      average='macro',  # 設定平均方式為 macro。
                      name='macro_f1'),  # 指標名稱。
            tf.keras.metrics.SparseCategoricalAccuracy()]  # 稀疏類別準確率。
  elif task_type == 'multi_label':  # 多標籤任務分支。
    return [tf.keras.metrics.Precision(name='precision'),  # 精確率指標。
            tf.keras.metrics.AUC(curve='PR',  # PR 曲線下面積。
                                 multi_label=True,  # 啟用多標籤模式。
                                 num_labels=num_classes,  # 指定標籤數量。
                                 from_logits=True,  # 指出模型輸出為 logits。
                                 name='pr_auc')]  # 指標名稱。
  else:  # 任務型態不支援分支。
    raise ValueError  # 拋出錯誤提醒。


def get_early_stopping_callback(  # 定義函式：建立 EarlyStopping callback。
    monitor_metric='val_loss',  # 監控指標預設為驗證損失。
    start_from_epoch=20,  # 從第幾個 epoch 後才開始監控停訓。
    patience=5,  # 容忍幾個 epoch 未改善。
    verbose=1,  # 輸出訊息等級。
    mode='auto',  # 自動判斷指標是要最小化或最大化。
    restore_best_weights=True):  # 是否回復最佳權重。

  return tf.keras.callbacks.EarlyStopping(  # 回傳 EarlyStopping 實例。
      monitor=monitor_metric,  # 套用監控指標。
      start_from_epoch=start_from_epoch,  # 套用開始監控 epoch。
      patience=patience,  # 套用 patience。
      verbose=verbose,  # 套用輸出等級。
      mode=mode,  # 套用模式。
      restore_best_weights=restore_best_weights,  # 套用回復最佳權重設定。
  )  # 結束 callback 建立。


def get_checkpoint_callback(  # 定義函式：建立模型檢查點 callback。
    exp_dir,  # 實驗輸出資料夾。
    monitor='val_loss',  # 監控指標。
    verbose=1,  # 輸出訊息等級。
    save_best_only=True,  # 僅儲存最佳模型。
    save_weights_only=False,  # 是否只存權重。
    mode='auto',  # 指標比較模式。
    save_freq='epoch',  # 儲存頻率。
):
  checkpoint_string = '/checkpoints/epoch_{epoch:02d}'  # 定義 checkpoint 檔名模板。
  return tf.keras.callbacks.ModelCheckpoint(  # 回傳 ModelCheckpoint 實例。
      exp_dir + checkpoint_string,  # 設定儲存路徑。
      monitor=monitor,  # 套用監控指標。
      verbose=verbose,  # 套用輸出等級。
      save_best_only=save_best_only,  # 套用是否僅存最佳。
      save_weights_only=save_weights_only,  # 套用是否只存權重。
      mode=mode,  # 套用模式。
      save_freq=save_freq,  # 套用儲存頻率。
  )  # 結束 callback 建立。


def get_tensorboard_callback(exp_dir):  # 定義函式：建立 TensorBoard callback。
  return tf.keras.callbacks.TensorBoard(  # 回傳 TensorBoard 實例。
      log_dir=exp_dir + '/logs',  # 設定 log 輸出資料夾。
      write_graph=False,  # 不寫入計算圖以減少檔案量。
      write_steps_per_second=True,  # 啟用每秒步數統計。
      update_freq='epoch')  # 每個 epoch 更新一次。


def get_reduce_lr_plateau_callback(  # 定義函式：建立學習率下降 callback。
    monitor='val_loss',  # 監控指標。
    factor=0.3,  # 學習率衰減倍率。
    patience=10,  # 幾個 epoch 無改善後觸發。
    verbose=1,  # 輸出訊息等級。
    mode='auto',  # 指標比較模式。
    min_lr=1e-5,  # 學習率下限。
):
  return tf.keras.callbacks.ReduceLROnPlateau(  # 回傳 ReduceLROnPlateau 實例。
      monitor=monitor,  # 套用監控指標。
      factor=factor,  # 套用衰減倍率。
      patience=patience,  # 套用 patience。
      verbose=verbose,  # 套用輸出等級。
      mode=mode,  # 套用模式。
      min_lr=min_lr,  # 套用最小學習率。
  )  # 結束 callback 建立。


def get_learning_rate_step_scheduler_callback(  # 定義函式：建立階梯式學習率排程 callback。
    learning_rate=1e-4,  # 初始學習率參數（供介面一致性使用）。
    factor=0.3,  # 每次命中里程碑時的乘法係數。
    milestones=[30],  # 觸發衰減的 epoch 清單。
    verbose=1,  # 輸出訊息等級。
):
  def scheduler(epoch, learning_rate):  # 內部排程函式：每個 epoch 計算新學習率。
    if epoch in milestones:  # 若當前 epoch 命中里程碑。
      return learning_rate * factor  # 回傳衰減後學習率。
    else:  # 若未命中里程碑。
      return learning_rate  # 保持原學習率不變。

  return tf.keras.callbacks.LearningRateScheduler(  # 回傳 LearningRateScheduler 實例。
      scheduler,  # 指定排程函式。
      verbose=verbose,  # 套用輸出等級。
  )  # 結束 callback 建立。


def get_callbacks(callbacks_names, exp_dir, monitor_metric, learning_rate):  # 定義函式：依名稱集合組裝 callback 清單。
  callbacks = []  # 初始化空清單。
  if 'checkpoint' in callbacks_names:  # 若需要 checkpoint callback。
    callbacks.append(get_checkpoint_callback(exp_dir, monitor_metric))  # 加入 checkpoint callback。
  if 'reduce_lr_plateau' in callbacks_names:  # 若需要 Plateau 降學習率 callback。
    callbacks.append(get_reduce_lr_plateau_callback(monitor_metric))  # 加入 ReduceLROnPlateau callback。
  if 'step_scheduler' in callbacks_names:  # 若需要階梯式學習率排程 callback。
    callbacks.append(get_learning_rate_step_scheduler_callback(  # 加入 LearningRateScheduler callback。
        learning_rate=learning_rate))  # 傳入當前學習率設定。
  if 'early_stopping' in callbacks_names:  # 若需要 EarlyStopping callback。
    callbacks.append(get_early_stopping_callback())  # 加入 EarlyStopping callback。
  if 'tensorboard' in callbacks_names:  # 若需要 TensorBoard callback。
    callbacks.append(get_tensorboard_callback(exp_dir))  # 加入 TensorBoard callback。
  return callbacks  # 回傳組裝完成的 callback 清單。
