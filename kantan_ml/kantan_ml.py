import os, io, shutil
import random, string
import numpy as np
import tensorflow as tf
import urllib.request
from PIL import Image
from typing import List, Optional, Tuple
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import DesiredCapabilities
import time, glob
from .kanatan_model import KantanModel, KerasModel

class KantanML:
    
    def __init__(self, chromedriver_path: Optional[str] = None,
                 output_dir: str = os.path.abspath('.'), 
                 is_all_clean: bool = False,) -> None:
        '''
        クラスの初期化

        Parameters
        ----------
        chromedriver_path: str
            chromedriverのパスを指定
        output_dir: str
            ここでは画像やモデルを出力する際のディレクトリの指定を行う。
            ここで指定したディレクトリ直下にoutputsディレクトリが生成される
        is_all_clean: bool (Default: False)
            毎回初期化の処理を行うか否か？
        '''

        if chromedriver_path is None: 
            print('Kanatan_ML: 引数chromedriver_path が None の場合データセットの作成を行えません')

        if chromedriver_path[:4] == 'http':
            options = webdriver.ChromeOptions()
            options.binary_location = '/usr/bin/google-chrome'
            options.add_argument('--no-sandbox')
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            capabilities = options.to_capabilities()
            self.driver = webdriver.Remote(command_executor=chromedriver_path,
                                           desired_capabilities=capabilities)
        else:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--start-fullscreen')
            options.add_argument('--disable-plugins')
            options.add_argument('--disable-extensions')

            self.driver = webdriver.Chrome(chromedriver_path, options=options)
        
        self.output_dir: str       = os.path.join(output_dir, 'outputs')
        self.label_output_dir: str = os.path.join(self.output_dir, 'labels')
        self.model_output_dir: str = os.path.join(self.output_dir, 'models')
        self.log_output_dir: str   = os.path.join(self.output_dir, 'log')
        self._init_output_dir(is_clean_dir = is_all_clean)
    
    def __del__(self):
        self.driver.close()
        self.driver.quit()

    def _init_dir(self, dir_path, is_clean_dir: bool = False) -> None:
        '''
        ディレクトリの初期化

        Parameters
        ----------
        dir_path: str
            ディレクトリのパス
        is_clean_dir: bool (Default: False)
            もしディレクトリがすでに存在する場合その中身を削除するか？
        '''
        
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        elif is_clean_dir :
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
    
    def _init_output_dir(self, is_clean_dir: bool = False) -> None:
        '''
        outputsディレクトリの初期化

        Parameters
        ----------
        is_clean_dir: bool (Default: False)
            もしディレクトリがすでに存在する場合その中身を削除するか？
        '''

        self._init_dir(self.output_dir, is_clean_dir=is_clean_dir)
        self._init_dir(self.label_output_dir, is_clean_dir=is_clean_dir)
        self._init_dir(self.model_output_dir, is_clean_dir=is_clean_dir)
        self._init_dir(self.log_output_dir, is_clean_dir=is_clean_dir)
    
    def _save_image(self, img_url: str, save_png_path: str) -> None:
        '''
        urlから画像を255x255で保存する

        Parameters
        ----------
        img_url: str
            画像のURL
        save_png_path: str
            保存する画像のパス(ファイル名)
        '''

        f = io.BytesIO(urllib.request.urlopen(img_url).read())
        img = Image.open(f)
        img = img.resize((256, 256))
        img.save(save_png_path)
    
    def _randomname(self, n):
        randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
        return ''.join(randlst) + '_'

    def get_serch_image_url(self, keyword: str, start_number: int = 1) -> str:
        '''
        Yahoo！画像検索を用いたURLの作成

        Parameters
        ----------
        keyword: str
            検索ワード(例:"海老")を書く
        start_number: int (Default: 1)
            検索ページの最初の画像を実際の検索の何番目にするか？
        
        Returns
        -------
        str
            Yahoo!画像検索のURL
        '''

        return f'https://search.yahoo.co.jp/image/search?p={keyword}&ei=UTF-8&b={start_number}'
    
    def download_images_url_once(self, keyword: str, start_number: int, wait_time: int = 5) -> Tuple[List[str], int]:
        '''
        １ページ分の画像のURLを取得

        Parameters
        ----------
        keyword: str
            検索ワード(例:"海老")を書く
        start_number: 
            検索ページの最初の画像を実際の検索の何番目にするか？
        
        Returns
        -------
        List[str]
            画像のURLの一覧
        int
            表示された画像の枚数
            (画像のURLの一覧の長さではないことに注意)
        '''

        self.driver.get( self.get_serch_image_url(keyword, start_number=start_number) )
        start_time = time.time()
        while True:
            try :
                self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
                time.sleep(.5)
                if time.time() - start_time >= wait_time: break
            except:
                raise Exception('selenium error.')
        img_elms = self.driver.find_elements_by_tag_name('img')
        img_urls = [ elm.get_attribute('src') for elm in img_elms ]
        load_count = start_number + len(img_urls) - 1
        img_urls = [*filter(lambda u: u[:4]=='http' and u.split('.')[-1].lower() in ['png', 'jpeg', 'jpg'], img_urls)]
        return img_urls, load_count

    def download_images_url(self, keyword: str, want_number: int) -> List[str]:
        '''
        １ラベルの画像を必要以上にダウンロードする。

        Parameters
        ----------
        keyword: str
            検索ワード(例:"海老")を書く
        want_number: int
            最低限、欲しい画像の枚数を書く
        
        Returns
        -------
        List[str]
            画像のURLの一覧
        '''

        num = 1
        image_urls = []
        while True:
            _img_urls, _load_count = self.download_images_url_once(keyword, num)
            image_urls.extend(_img_urls)
            num += _load_count
            if len(image_urls) >= want_number: break
        return image_urls

    def download_images(self, search_words: List[str], number: int, is_clean_dir: bool = True) -> None:
        '''
        データセットの作成

        Parameters
        ----------
        search_words: List[str]
            検索単語のリスト
        number: int
            ラベルごとの画像枚数
        is_clean_dir: bool (Default: True)
            ラベルの中身を削除する
        '''

        # TODO: Twitterとかpixivとかからも持ってきたい

        rand_prefix = self._randomname(5)
        for label in search_words:
            label_urls = self.download_images_url(label, number)[:number]
            label_dir = os.path.join(self.label_output_dir, label)
            self._init_dir(label_dir, is_clean_dir=is_clean_dir)
            for i, url in enumerate(label_urls):
                img_path = os.path.join(label_dir, rand_prefix + (f'{i+1}'.zfill(5)) + '.png')
                self._save_image(url, img_path)
                print('saved:', img_path)
                time.sleep(random.random()/2)
    
    def get_model(self, model_name: str = 'mini_dnn', use_imagenet: bool = True, 
                  is_freeze: bool = True, fc_hidden: int = 128) -> KerasModel :
        '''
        モデルの作成

        Parameters
        ----------
        model_name: str (Default: "mini_dnn")
            どの深層学習モデルを使うか？
            mini_dnn: 簡単なDNN
            mini_cnn: 簡単なCNN
            efficientnetb0: つよつよモデル
        use_imagenet: bool (Default: True)
            学習済みモデルを用いるか？
            mini_dnn, mini_cnn では使えない
        is_freeze: bool (Default: True)
            use_imagenetがTrueの際に
            畳み込み層の重みを固定するか？
        fc_hidden: int (Default: 128)
            fc層の隠れ層のユニット数
        
        Returns
        -------
        KerasModel
            Kerasのモデル
        '''

        class_num = len(glob.glob(os.path.join(self.label_output_dir, '*')))

        if model_name == 'efficientnetb0':
            base_model = None
            if use_imagenet:
                base_model = tf.keras.applications.EfficientNetB0(input_shape=(256, 256, 3), weights='imagenet', include_top=False)
                if is_freeze:
                    base_model.trainable = False
            else :
                base_model = tf.keras.applications.EfficientNetB0(input_shape=(256, 256, 3), include_top=False)
            
            return tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(fc_hidden, activation='relu'),
                tf.keras.layers.Dropout(.3),
                tf.keras.layers.Dense(class_num, activation='softmax'),
            ])

        if model_name == 'mini_dnn':
            return tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(256, 256, 3)),
                tf.keras.layers.Dense(fc_hidden, activation='relu'),
                tf.keras.layers.Dropout(.3),
                tf.keras.layers.Dense(class_num, activation='softmax'),
            ])
        
        if model_name == 'mini_cnn':
            return tf.keras.Sequential([
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(fc_hidden, activation='relu'),
                tf.keras.layers.Dropout(.3),
                tf.keras.layers.Dense(class_num, activation='softmax'),
            ])
    
    def get_datasets(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        データセットのロード

        Returns
        ----------
        np.ndarray
            画像のデータセット
        np.ndarray
            ラベルのデータセット
        '''

        labels_path = sorted(glob.glob(os.path.join(self.label_output_dir, '*')))
        x = []
        y = []
        for i, lp in enumerate(labels_path):
            for ip in glob.glob(os.path.join(lp, '*.png')):
                img = tf.keras.preprocessing.image.load_img(ip, target_size=(256, 256, 3))
                img = tf.keras.preprocessing.image.img_to_array(img) / 255.
                lab = tf.keras.utils.to_categorical(i, len(labels_path))
                x.append(img)
                y.append(lab)
        return np.array(x), np.array(y)

    def train(self, model: KerasModel, epochs: int = 100, batch_size: int = 8, 
              patience: int = 2, is_save_model: bool = True) -> KerasModel:
        '''
        モデルのコンパイル＆学習
        
        Parameters
        ----------
        model: KerasModel
            Kerasモデル
        epochs: int (Default: 100)
            エポック数
        batch_size: int (Default: 8)
            バッチサイズ
        patience: int (Default: 2)
            トレーニングが停止されるまで改善されていないエポックの数
        is_save_model: bool (Default: True)
            モデルを保存するか？
        '''
        
        # TODO: かさ増しを行いたい

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=patience
        )
        tensor_bord = tf.keras.callbacks.TensorBoard(log_dir=self.log_output_dir, histogram_freq=1)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        x, y = self.get_datasets()
        model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, callbacks=[early_stopping, tensor_bord])
        if is_save_model: model.save(os.path.join(self.model_output_dir, f'model_{int(time.time())}.h5'))
        return model
    
    def load_model(self, h5_path: str) -> KerasModel :
        '''
        モデルのロード

        Parameters
        ----------
        h5_path: str
            h5ファイル(深層学習モデル)のロード
        
        Returns
        -------
        KerasModel
            Kreasのモデル
        '''

        return tf.keras.models.load_model(h5_path)
    
    def inference(self, model: KerasModel, img_path: str) -> str:
        '''
        モデルを用いて画像の推論
        
        Parameters
        ----------
        model: KerasModel
            Kerasモデル
        img_path: str
            推論する画像のパス
        
        Returns
        -------
        str
            推論した結果のラベル
        '''

        labels = [*map(lambda p:p.split('/')[-1], sorted(glob.glob(os.path.join(self.label_output_dir, '*'))))]
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256, 3))
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.
        img = img.reshape(1, 256, 256, -1)
        out = labels[model.predict(img)[0].argmax()]
        return out
    
    def inference_for_url(self, model: KerasModel, img_url: str):
        '''
        モデルを用いてインターネットの画像の推論
        
        Parameters
        ----------
        model: KerasModel
            Kerasモデル
        img_path: str
            推論する画像のURL(http://xxxx/xxxx.pngなど)
        
        Returns
        -------
        str
            推論した結果のラベル
        '''

        labels = [*map(lambda p:p.split('/')[-1], sorted(glob.glob(os.path.join(self.label_output_dir, '*'))))]
        f = io.BytesIO(urllib.request.urlopen(img_url).read())
        img = Image.open(f)
        img = img.resize((256, 256))
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.
        img = img.reshape(1, 256, 256, -1)
        out = labels[model.predict(img)[0].argmax()]
        return out
    
    def keras_model_to_kantan_model(self, model: KerasModel) -> KantanModel:
        labels = [*map(lambda p:p.split('/')[-1], sorted(glob.glob(os.path.join(self.label_output_dir, '*'))))]
        km = KantanModel(model, labels)
        return km