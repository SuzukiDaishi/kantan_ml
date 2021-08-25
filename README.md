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