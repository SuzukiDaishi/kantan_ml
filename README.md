# kantan_ml

## 事前準備
`chromedriver`をインストール

### Dockerでどっかーん！！の場合
```bash
docker run -d -p 4444:4444 selenium/standalone-chrome
```

## pip install
```bash
pip install git+https://github.com/SuzukiDaishi/kantan_ml
```

## 使用例
```python
from kantan_ml import KantanML

# ドライバを設定 & 初期化
ml = KantanML('/path/to/chromedriver')

# Dockerでstandalone-chromeを用意した場合
# ml = KantanML('http://localhost:4444/wd/hub')

# データセットの作成(スクレイピング)
ml.download_images(['月ノ美兎', '樋口楓', '静凛'], 5)

# 深層学習モデルの設定
model = ml.get_model('mini_cnn')

# モデルの学習
model = ml.train(model)

# モデルを使いやすい形に
kmodel = ml.keras_model_to_kantan_model(model)


# 画像を推論
out = kmodel.inference_for_url_probability('http://url/to/image/path.png')

# 結果を確認
for k in out:
    print(k, ':', out[k])

```

## 単体で使う場合

```bash

# gitからcloneする
git clone https://github.com/SuzukiDaishi/kantan_ml.git
cd kantan_ml

# データセット作成 & 学習します
# --labels: 学習するラベルを複数記載する
# --save-dir: 保存するディレクトリ+保存名をフルパスで記載する
python example.py train --labels 月ノ美兎 樋口楓 静凛 --save-dir /path/to/dir/model_name

# 推論する
# --labels: train時と同様
# --save-dir: train時と同様
# --images-url: 推論する画像のURLを表示する
python example.py inference --labels 月ノ美兎 樋口楓 静凛 --save-dir /path/to/dir/model_name --images-url http://path/to/img.png http://path/to/img.jpg https://path/to/img2.png
```