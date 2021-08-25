from typing import Dict, List, Optional, Tuple, Union
import tensorflow as tf
import urllib.request
from PIL import Image
import os, io, shutil

KerasModel = Union[tf.keras.Model, tf.keras.Sequential]

class KantanModel:

    def __init__(self, model: Union[KerasModel, str], labels: List[str]) -> None:
        if type(model) is str:
            self.model = tf.keras.models.load_model(model)
        else:
            self.model = model
        self.labels = labels
    
    def inference(self, img_path: str) -> str:
        '''
        モデルを用いて画像の推論
        
        Parameters
        ----------
        img_path: str
            推論する画像のパス
        
        Returns
        -------
        str
            推論した結果のラベル
        '''

        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256, 3))
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.
        img = img.reshape(1, 256, 256, -1)
        out = self.labels[self.model.predict(img)[0].argmax()]
        return out
    
    def inference_probability(self, img_path: str) -> Dict[str, float]:
        '''
        モデルを用いて画像の推論
        
        Parameters
        ----------
        img_path: str
            推論する画像のパス
        
        Returns
        -------
        Dict[str, float]
            ラベルと推論した確率
        '''

        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256, 3))
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.
        img = img.reshape(1, 256, 256, -1)
        probs = self.model.predict(img)[0]
        out = {}
        for i in range(probs.shape[0]):
            out[ self.labels[i] ] = float(probs[i])
        return out

    def inference_for_url(self, img_url: str) -> str:
        '''
        モデルを用いてインターネットの画像の推論
        
        Parameters
        ----------
        img_path: str
            推論する画像のURL(http://xxxx/xxxx.pngなど)
        
        Returns
        -------
        str
            推論した結果のラベル
        '''

        f = io.BytesIO(urllib.request.urlopen(img_url).read())
        img = Image.open(f).convert('RGB')
        img = img.resize((256, 256))
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.
        img = img.reshape(1, 256, 256, -1)
        out = self.labels[self.model.predict(img)[0].argmax()]
        return out
    
    def inference_for_url_probability(self, img_url: str) -> Dict[str, float]:
        '''
        モデルを用いてインターネットの画像の推論
        
        Parameters
        ----------
        img_path: str
            推論する画像のURL(http://xxxx/xxxx.pngなど)
        
        Returns
        -------
        Dict[str, float]
            ラベルと推論した確率
        '''

        f = io.BytesIO(urllib.request.urlopen(img_url).read())
        img = Image.open(f).convert('RGB')
        img = img.resize((256, 256))
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.
        img = img.reshape(1, 256, 256, -1)
        probs = self.model.predict(img)[0]
        out = {}
        for i in range(probs.shape[0]):
            out[ self.labels[i] ] = float(probs[i])
        return out