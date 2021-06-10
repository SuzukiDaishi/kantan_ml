import os, shutil
from typing import List

class Kanatan_ML:
    
    def __init__(self, output_dir: str = os.path.abspath('.'), is_all_clean: bool = False) -> None:
        '''
        クラスの初期化

        Parameters
        ----------
        output_dir: str
            ここでは画像やモデルを出力する際のディレクトリの指定を行う。
            ここで指定したディレクトリ直下にoutputsディレクトリが生成される
        is_all_clean: bool (Default: False)
            毎回初期化の処理を行うか否か？
        '''
        
        self.output_dir: str       = os.path.join(output_dir, 'outputs')
        self.label_output_dir: str = os.path.join(self.output_dir, 'labels')
        self.model_output_dir: str = os.path.join(self.output_dir, 'models')
        self._init_output_dir(is_clean_dir=is_all_clean)

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

    def download_images(self, search_words: List[str], number: int) -> str:
        pass