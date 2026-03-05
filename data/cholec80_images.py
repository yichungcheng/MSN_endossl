'''Module for creating TF datasets for Cholec80 dataset'''  # 這個模組用於建立 Cholec80 影像資料集的 TensorFlow Dataset。

import os  # 匯入作業系統路徑工具。
import tensorflow as tf  # 匯入 TensorFlow 主套件。
import tensorflow_models as tfm  # 匯入 TensorFlow Models，這裡用來做資料增強。


_CHOLEC80_PHASES_WEIGHTS = {  # 定義各手術階段類別的權重（通常用於不平衡資料訓練）。
    0: 1.9219914802981897,  # 類別 0 的權重。
    1: 0.19571110990619747,  # 類別 1 的權重。
    2: 0.9849911311229362,  # 類別 2 的權重。
    3: 0.2993075998175712,  # 類別 3 的權重。
    4: 1.942680301399354,  # 類別 4 的權重。
    5: 1.0,  # 類別 5 的權重。
    6: 2.2015858493443123  # 類別 6 的權重。
    }  # 權重字典結束。


_LABEL_NUM_MAPPING = {  # 定義文字標籤到數字類別 ID 的對應。
    'GallbladderPackaging': 0,  # 膽囊包裝階段對應到 0。
    'CleaningCoagulation': 1,  # 清潔凝固階段對應到 1。
    'GallbladderDissection': 2,  # 膽囊剝離階段對應到 2。
    'GallbladderRetraction': 3,  # 膽囊牽引階段對應到 3。
    'Preparation': 4,  # 準備階段對應到 4。
    'ClippingCutting': 5,  # 夾閉與切割階段對應到 5。
    'CalotTriangleDissection': 6  # Calot 三角剝離階段對應到 6。
    }  # 標籤映射字典結束。


_SUBSAMPLE_RATE = 25  # 標註序列取樣間隔（每 25 個標註取 1 個）。


_CHOLEC80_SPLIT = {'train': range(1, 41),  # 訓練集使用 video01 ~ video40。
                   'validation': range(41, 49),  # 驗證集使用 video41 ~ video48。
                   'test': range(49, 81)}  # 測試集使用 video49 ~ video80。


curr_dir = os.path.dirname(os.path.realpath(__file__))  # 取得目前檔案所在資料夾絕對路徑。
config_path = os.path.join(curr_dir, 'config.json')  # 組出 config.json 的完整路徑。


resize = tf.keras.layers.Resizing(224, 224, crop_to_aspect_ratio=False)  # 建立固定縮放到 224x224 的層（不保持長寬比裁切）。


resize_and_center_crop = tf.keras.layers.Resizing(  # 建立縮放到 224x224 並保持比例中心裁切的層。
    224, 224, crop_to_aspect_ratio=True  # 設定輸出尺寸與裁切行為。
)


_RAND_AUGMENT = tfm.vision.augment.RandAugment(  # 建立 RandAugment 物件做影像資料增強。
    num_layers=3, magnitude=7, exclude_ops=['Invert', 'Solarize', 'Posterize']  # 設定增強層數、強度與排除的操作。
)


def randaug(image):  # 定義使用 RandAugment 的影像轉換函式。
  image = resize(image)  # 先將影像縮放到固定尺寸。
  return _RAND_AUGMENT.distort(image * 255.0) / 255.0  # 轉成 0~255 做增強後再縮回 0~1。


def get_train_image_transformation(name):  # 根據名稱回傳訓練時要使用的影像轉換函式。
  if name == 'randaug':  # 若指定 randaug。
    return randaug  # 回傳含 RandAugment 的處理函式。
  else:  # 其他情況。
    return resize  # 回傳只有縮放的處理函式。


class Cholec80ImagesLoader:  # 封裝 Cholec80 影像與標籤讀取流程的類別。
    def __init__(self, data_root, video_ids, batch_size, shuffle=False, augment=resize):  # 初始化資料載入器。
        self.batch_size = batch_size  # 儲存 batch 大小。
        self.video_ids = video_ids  # 儲存要讀取的影片 ID 清單。
        self.data_root = data_root  # 儲存資料根目錄。
        self.batch_size = batch_size  # 再次設定 batch 大小（與上行重複）。
        self.shuffle = shuffle  # 是否在資料集上啟用 shuffle。
        self.augment = augment  # 儲存影像增強/前處理函式。

        self.all_frame_names, self.all_labels  = self.prebuild(video_ids)  # 預先整理所有影格路徑與對應標籤。

    def prebuild(self, video_ids):  # 掃描指定影片，建立影格檔案清單與標籤清單。
        frames_dir = os.path.join(self.data_root, 'frames')  # frames 資料夾路徑。
        annos_dir = os.path.join(self.data_root, 'phase_annotations')  # 相位標註檔資料夾路徑。

        all_labels = []  # 初始化總標籤列表。
        all_frame_names = []  # 初始化總影格路徑列表。
        for video_id in video_ids:  # 逐一處理每部影片。
            video_frames_dir = os.path.join(frames_dir, video_id)  # 當前影片影格資料夾路徑。
            frames = [os.path.join(video_frames_dir, f) for f in os.listdir(video_frames_dir)]  # 收集當前影片所有影格路徑。
            with open(os.path.join(annos_dir, video_id + '-phase.txt'), 'r') as f:  # 開啟當前影片的相位標註文字檔。
                labels = f.readlines()[1:]  # 讀取所有行並跳過標頭行。
            labels = [l.split('\t')[1][:-1] for l in labels]  # 取出每行第二欄位作為文字標籤並移除換行字元。
            labels = [_LABEL_NUM_MAPPING[l] for l in labels[::_SUBSAMPLE_RATE]][:len(frames)]  # 先依取樣率抽樣，再轉成數字標籤，並截到影格數量。
            all_frame_names += frames  # 將當前影片影格路徑加入總列表。
            all_labels += labels  # 將當前影片標籤加入總列表。
        return all_frame_names, all_labels  # 回傳完整影格與標籤列表。
    
    def parse_image(self, image_path):  # 將影像路徑解析成前處理後的 Tensor。
        img = tf.io.read_file(image_path)  # 讀取影像檔案內容（二進位）。
        img = tf.io.decode_jpeg(img, channels=3)  # 將 JPEG 解碼為 3 通道影像。
        return self.augment(img)  # 套用增強/縮放處理並回傳。

    def parse_label(self, label):  # 標籤解析函式。
        return label  # 目前直接原樣回傳標籤。

    def parse_example(self, image_path, label):  # 將單筆 (影像路徑, 標籤) 轉為模型可用格式。
        return (self.parse_image(image_path),  # 第一個輸出為處理後影像。
                self.parse_label(label))  # 第二個輸出為處理後標籤。
        
    def parse_example_image_path(self, image_path, label, image_path_):  # 版本二：額外保留原始影像路徑供追蹤。
        return (self.parse_image(image_path),  # 第一個輸出為處理後影像。
                self.parse_label(label),  # 第二個輸出為標籤。
                image_path_)  # 第三個輸出為影像路徑字串。

    def get_tf_dataset(self, with_image_path=False):  # 建立 tf.data.Dataset 管線。
        num_parallel_calls=tf.data.AUTOTUNE  # 使用 AUTOTUNE 自動決定平行處理數。

        ds_frames = tf.data.Dataset.list_files(self.all_frame_names, shuffle=False)  # 由影格路徑清單建立檔案資料集（不打亂）。
        ds_labels = tf.data.Dataset.from_tensor_slices(self.all_labels)  # 由標籤清單建立資料集。
        if with_image_path:  # 若需要在輸出中保留影像路徑。
            ds = tf.data.Dataset.zip((ds_frames, ds_labels, ds_frames))  # 將影像路徑、標籤與影像路徑本身打包。
            ds = ds.map(self.parse_example_image_path, num_parallel_calls=num_parallel_calls)  # 套用含路徑的解析函式。
        else:  # 若不需要輸出影像路徑。
            ds = tf.data.Dataset.zip((ds_frames, ds_labels))  # 僅打包影像路徑與標籤。
            ds = ds.map(self.parse_example, num_parallel_calls=num_parallel_calls)  # 套用一般解析函式。

        ds = ds.batch(self.batch_size)  # 依 batch_size 分批。
        if self.shuffle:  # 若設定要打亂。
            ds = ds.shuffle(1024)  # 以 buffer size 1024 進行隨機打亂。
        ds = ds.prefetch(num_parallel_calls)  # 預先抓取以提升輸送效率。
        return ds  # 回傳建立完成的資料集。


def get_cholec80_images_datasets(data_root, batch_size, train_transformation='randaug', with_image_path=False):  # 建立 train/validation/test 三個 split 的資料集。
    data = {}  # 初始化回傳字典。
    for split, ids_range in _CHOLEC80_SPLIT.items():  # 迭代每個資料切分與其影片編號範圍。
        if split == 'train':  # 若是訓練集。
            ds = Cholec80ImagesLoader(  # 建立 Cholec80ImagesLoader。
                data_root,  # 傳入資料根目錄。
                [f'video{i:02}' for i in ids_range],  # 將範圍轉成 videoXX 格式清單。
                batch_size,  # 傳入 batch 大小。
                augment=get_train_image_transformation(train_transformation)  # 設定訓練使用的影像轉換。
                )
        data[split] = ds.get_tf_dataset(with_image_path)  # 取得 TensorFlow Dataset 並放入對應 split 鍵值。
    return data  # 回傳所有 split 的資料集字典。


if __name__ == '__main__':  # 當此檔案被直接執行時才會進入此區塊。
    par_dir = os.path.realpath(__file__+ '/../../')  # 推算專案上層資料夾路徑。
    data_root = os.path.join(par_dir, 'cholec80', 'cholec80')  # 組合預設資料集根目錄。
    datasets = get_cholec80_images_datasets(data_root, 8)  # 建立 batch size=8 的資料集。
    for b in datasets['validation']:  # 迭代驗證集第一個 batch。
        print(b[0].shape)  # 印出影像張量形狀。
        print(b[1].shape)  # 印出標籤張量形狀。
        break  # 只看第一個 batch 後跳出迴圈。
