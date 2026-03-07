''Simple script for loading pre-trained model and extracting hidden representations from it.'''
# 檔案說明：此腳本示範如何載入已訓練好的模型，並抽取影像的隱藏特徵向量。

import tensorflow as tf  # 匯入 TensorFlow，負責影像前處理、模型載入與推論。
import argparse  # 匯入 argparse，用來解析命令列參數。


def get_image(image_path, size=224):  # 定義影像讀取函式：輸入影像路徑與目標尺寸。
    img = tf.keras.preprocessing.image.load_img(image_path)  # 從磁碟讀取影像檔，回傳 PIL Image。
    img = tf.convert_to_tensor(img)  # 將 PIL Image 轉成 TensorFlow Tensor，方便後續運算。
    return tf.image.resize(img, [size, size])  # 將影像縮放為 size x size，作為模型輸入。


class ModelWrapper(tf.keras.Model):  # 建立包裝類別，讓已儲存模型可用 Keras 介面呼叫。
  def __init__(self, model_path):  # 初始化：接收 SavedModel 的路徑。
    super().__init__()  # 呼叫父類別建構子，完成 Keras Model 基本初始化。
    self.backbone = tf.saved_model.load(model_path)  # 載入 SavedModel，存成 backbone 供 call 使用。

  def call(self, x):  # 定義前向傳播：輸入一個 batch 影像張量。
    return self.backbone(x)[1]  # 執行 backbone，並取回傳結果中的第 2 個元素作為嵌入向量。

  def get_config(self):  # 提供模型設定（序列化時可能使用）。
    return super().get_config()  # 直接沿用父類別的預設設定內容。


parser = argparse.ArgumentParser()  # 建立命令列參數解析器。

parser.add_argument(  # 新增 --model_path 參數。
    "--model_path",  # 參數名稱：模型路徑。
    type=str,  # 參數型別為字串。
    default='/root/vits_lapro_private/saved_model_inference',  # 預設模型路徑。
    help="Path to a pre-trained model",  # 參數說明文字。
)
parser.add_argument(  # 新增 --image_path 參數。
    "--image_path",  # 參數名稱：影像路徑。
    type=str,  # 參數型別為字串。
    default='/root/cholec80/samples/video01_000001.png',  # 預設影像路徑。
    help="Path to a image file",  # 參數說明文字。
)

def main(args):  # 主流程函式：接收解析後的命令列參數。
    model = ModelWrapper(args.model_path)  # 依指定路徑建立模型包裝器並載入模型。
    img = get_image(args.image_path)  # 讀取並縮放指定影像。
    embed = tf.squeeze(model(tf.expend_dims(img, 0)), 0)  # 增加 batch 維度後推論，再移除 batch 維度取得單張 embedding。
    print('Hidden vector shape is: {}'.format(embed.shape))  # 輸出隱藏向量的形狀資訊。


if __name__ == '__main__':  # 若此檔案是直接執行（非被 import）時。
    args = parser.parse_args()  # 解析命令列參數。
    main(args)  # 執行主流程。
