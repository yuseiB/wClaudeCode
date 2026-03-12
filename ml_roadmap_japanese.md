# 機械学習 学習ロードマップ（roadmap.sh 準拠）

> **参考元**: [roadmap.sh/machine-learning](https://roadmap.sh/machine-learning) / [roadmap.sh/ai-data-scientist](https://roadmap.sh/ai-data-scientist)
> **対象期間**: 約 9〜12 ヶ月（プログラミング経験者の場合 6 ヶ月も可能）
> **凡例**: 🆓 = 無料・オンライン公開あり　📖 = 有料書籍

---

## フェーズ 1: 数学・統計の基礎（1〜2 ヶ月目）

### 1-1. 線形代数
- ベクトル・行列の演算
- 行列式・逆行列
- 固有値・固有ベクトル（PCA の理解に必須）
- 特異値分解（SVD）

### 1-2. 微積分
- 微分・偏微分
- 勾配（gradient）と連鎖律（chain rule）
- 最適化手法の基礎（勾配降下法）

### 1-3. 確率・統計
- 確率分布（正規分布・ベルヌーイ分布・ポアソン分布）
- 条件付き確率・ベイズの定理
- 仮説検定・信頼区間
- 統計的推定

### 推薦教科書・リソース

| タイトル | 著者 | 入手 | 特徴 |
|---|---|---|---|
| **Mathematics for Machine Learning** | Deisenroth, Faisal, Ong | 🆓 [mml-book.github.io](https://mml-book.github.io/) | ML に必要な数学を一冊で網羅。英語だが公式 PDF 無料公開 |
| **メディカルAI専門コース 数学基礎** | 日本メディカルAI学会 | 🆓 [japan-medical-ai.github.io](https://japan-medical-ai.github.io/medical-ai-course-materials/notebooks/01_Basic_Math_for_ML.html) | 線形代数・微分・確率を Google Colab で実行しながら学べる日本語教材 |
| **東工大「機械学習帳」** | 東京工業大学 | 🆓 [chokkan.github.io/mlnote](https://chokkan.github.io/mlnote/) | 図・アニメーション豊富な日本語の機械学習講義ノート |
| **「統計学入門」（赤本）** | 東京大学教養学部統計学教室 | 📖 | 日本語定番統計テキスト。確率・推定・検定を体系的に学習 |

---

## フェーズ 2: プログラミング基礎（1〜2 ヶ月目・並行）

### 2-1. Python 基礎
- データ型・制御構文・関数・クラス
- ファイル入出力・例外処理

### 2-2. データサイエンス ライブラリ
- **NumPy**: 配列・行列演算
- **Pandas**: データフレーム・データ前処理
- **Matplotlib / Seaborn**: データ可視化

### 2-3. 開発環境
- Jupyter Notebook / Google Colab
- Git / GitHub によるバージョン管理

### 推薦教科書・リソース

| タイトル | 著者 | 入手 | 特徴 |
|---|---|---|---|
| **Python チュートリアル（公式）** | Python Software Foundation | 🆓 [docs.python.org/ja/3/tutorial](https://docs.python.org/ja/3/tutorial/) | 公式・完全無料・日本語あり。Python の基礎はここから |
| **東大松尾研「DL4US」** | 東京大学松尾研究室 | 🆓 [github.com/matsuolab/DL4US](https://github.com/matsuolab/DL4US) | NumPy/scikit-learn から深層学習まで対応の Jupyter ベース日本語教材 |
| **「Pythonによるデータ分析入門」** | Wes McKinney（著）、小林儀匡（訳） | 📖 | Pandas 作者自身が書いた定番実践書。日本語訳あり |
| **「Python機械学習プログラミング」** | Sebastian Raschka, Vahid Mirjalili | 📖 | scikit-learn を使った実践入門。日本語訳（第3版）あり |

---

## フェーズ 3: 機械学習コアアルゴリズム（3〜5 ヶ月目）

### 3-1. 教師あり学習（Supervised Learning）
- 線形回帰・ロジスティック回帰
- 決定木・ランダムフォレスト
- サポートベクターマシン（SVM）
- k 近傍法（k-NN）
- 勾配ブースティング（XGBoost, LightGBM）

### 3-2. 教師なし学習（Unsupervised Learning）
- クラスタリング（k-means, DBSCAN）
- 次元削減（PCA, t-SNE, UMAP）
- 異常検知

### 3-3. 強化学習（Reinforcement Learning）入門
- マルコフ決定過程（MDP）
- Q学習・方策勾配法

### 3-4. モデル評価
- 交差検証・過学習・正則化
- 混同行列・精度・再現率・F1スコア
- ROC 曲線・AUC
- バイアス・バリアンストレードオフ

### 3-5. データ前処理
- 欠損値処理・外れ値処理
- 特徴量エンジニアリング
- 正規化・標準化
- データ拡張

### 推薦教科書・リソース

| タイトル | 著者 | 入手 | 特徴 |
|---|---|---|---|
| **パターン認識と機械学習の学習 普及版** | 光成滋生 | 🆓 [herumi.github.io/prml](https://herumi.github.io/prml/) | PRML（Bishop本）の日本語解説 PDF。CC ライセンスで無料公開 |
| **awesome-prml-ja（PRML 解説資料集）** | tsg-ut（有志） | 🆓 [github.com/tsg-ut/awesome-prml-ja](https://github.com/tsg-ut/awesome-prml-ja) | PRML に関する日本語の勉強会・読書会資料を集約した GitHub リポジトリ |
| **Interpretable Machine Learning 日本語版** | Christoph Molnar（著）、HACARUS（訳） | 🆓 [hacarus.github.io/interpretable-ml-book-ja](https://hacarus.github.io/interpretable-ml-book-ja/) | モデルの解釈可能性を扱う名著の非公式日本語訳。無料公開 |
| **「scikit-learnとTensorFlowによる実践機械学習」** | Aurélien Géron（著）、下田倫大（訳） | 📖 | 最も有名な ML 実践書。第3版まで日本語訳あり |
| **「パターン認識と機械学習」（PRML）** | Christopher M. Bishop（著）、元田浩ほか（訳） | 📖（英語原書は 🆓 [Microsoft Research](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)） | ML の聖典。英語原書 PDF は Microsoft Research で無料公開。日本語訳は有料書籍（上下2巻） |

---

## フェーズ 4: 深層学習（5〜7 ヶ月目）

### 4-1. ニューラルネットワーク基礎
- パーセプトロン・多層パーセプトロン（MLP）
- 活性化関数（ReLU, Sigmoid, Softmax）
- 誤差逆伝播法（Backpropagation）
- バッチ正規化・ドロップアウト

### 4-2. 畳み込みニューラルネットワーク（CNN）
- 畳み込み層・プーリング層
- LeNet, AlexNet, VGG, ResNet, EfficientNet
- 転移学習・ファインチューニング

### 4-3. 再帰型ニューラルネットワーク（RNN）
- 基本 RNN・勾配消失問題
- LSTM（長短期記憶）
- GRU（ゲート付き再帰ユニット）

### 4-4. Transformer とアテンション機構
- Self-Attention・Multi-Head Attention
- 「Attention is All You Need」論文
- BERT・GPT アーキテクチャ

### 4-5. 深層学習フレームワーク
- **PyTorch**: 研究・最先端向け（推奨）
- **TensorFlow / Keras**: プロダクション向け

### 推薦教科書・リソース

| タイトル | 著者 | 入手 | 特徴 |
|---|---|---|---|
| **Deep Learning（Goodfellow）** | Goodfellow, Bengio, Courville | 🆓 [deeplearningbook.org](https://www.deeplearningbook.org/) | 深層学習の定番教科書（通称 GBC 本）。英語だが全文無料公開。日本語訳書籍あり |
| **愛媛大学「深層学習の基礎と演習」** | 二宮崇（愛媛大学） | 🆓 [aiweb.cs.ehime-u.ac.jp/.../deeplearning.pdf](https://aiweb.cs.ehime-u.ac.jp/~ninomiya/enpitpro/deeplearning.pdf) | 「ゼロから作るDL」準拠の日本語講義資料 PDF。PyTorch 基礎も含む |
| **東大「DL4US」** | 東京大学松尾研究室 | 🆓 [github.com/matsuolab/DL4US](https://github.com/matsuolab/DL4US) | CNN・RNN・Attention まで扱う日本語 Jupyter 教材 |
| **「ゼロから作るDeep Learning」シリーズ** | 斎藤康毅 | 📖（コード: 🆓 [github.com/oreilly-japan/deep-learning-from-scratch](https://github.com/oreilly-japan/deep-learning-from-scratch)） | Python で一から実装。日本語オリジナル。全5巻シリーズ（①〜⑤） |
| **「深層学習」（機械学習プロフェッショナルシリーズ）** | 岡谷貴之 | 📖 | 数理的な深層学習の標準日本語テキスト（講談社） |

---

## フェーズ 5: 専門分野（7〜9 ヶ月目）

### 5-1. 自然言語処理（NLP）
- テキスト前処理（トークン化・ステミング・見出し語化）
- 単語埋め込み（Word2Vec, GloVe, FastText）
- Transformer ベースモデル（BERT, GPT, T5）
- 感情分析・機械翻訳・テキスト生成
- 大規模言語モデル（LLM）・プロンプトエンジニアリング

### 5-2. コンピュータビジョン（CV）
- 物体検出（YOLO, Faster R-CNN）
- セグメンテーション（U-Net, Mask R-CNN）
- 画像生成（GAN, Stable Diffusion）
- OpenCV による画像処理

### 5-3. 生成 AI（Generative AI）
- GAN（敵対的生成ネットワーク）
- 拡散モデル（Diffusion Models）
- VAE（変分オートエンコーダ）
- LLM のファインチューニング（LoRA, PEFT）

### 推薦教科書・リソース

| タイトル | 著者 | 入手 | 特徴 |
|---|---|---|---|
| **キカガク NLP 無料チュートリアル** | KIKAGAKU | 🆓 [free.kikagaku.ai/tutorial/basic_of_nlp](https://free.kikagaku.ai/tutorial/basic_of_nlp/) | テキスト前処理から機械学習 NLP までを日本語で解説。無料オンライン |
| **Hugging Face NLP Course（日本語版）** | Hugging Face | 🆓 [huggingface.co/learn/nlp-course/ja](https://huggingface.co/learn/nlp-course/ja/chapter1/1) | Transformers ライブラリを使った NLP コース。日本語翻訳あり、無料 |
| **「ゼロから作るDeep Learning ❷」（NLP編）** | 斎藤康毅 | 📖（コード: 🆓 [github.com/oreilly-japan/deep-learning-from-scratch-2](https://github.com/oreilly-japan/deep-learning-from-scratch-2)） | Word2Vec・RNN・Attention を一から実装。日本語オリジナル |
| **「自然言語処理の基礎」** | 岡崎直観, 荒瀬由紀, 鈴木潤ほか | 📖 | 日本語 NLP の大学院標準テキスト（オーム社、2022年） |
| **「Transformerによる自然言語処理」** | Lewis Tunstall ほか（著）、中山光樹（訳） | 📖 | Hugging Face 公式著者による実践書。日本語訳あり |

---

## フェーズ 6: MLOps と本番環境（10〜11 ヶ月目）

### 6-1. モデルのデプロイ
- REST API 作成（Flask, FastAPI）
- Docker によるコンテナ化
- クラウドデプロイ（AWS SageMaker, GCP Vertex AI, Azure ML）

### 6-2. MLOps
- 実験管理（MLflow, Weights & Biases）
- CI/CD パイプライン
- モデルモニタリング・ドリフト検知
- データバージョニング（DVC）

### 6-3. データエンジニアリング基礎
- データパイプライン（Apache Airflow）
- 特徴量ストア
- ビッグデータ（Spark 入門）

### 推薦教科書・リソース

| タイトル | 著者 | 入手 | 特徴 |
|---|---|---|---|
| **MLOps Yearning（楽天テック公開スライド）** | 楽天技術者 | 🆓 [slideshare.net/.../mlops-yearning](https://www.slideshare.net/rakutentech/mlops-yearning) | 実運用の ML システムを構築する前に考えること。日本語スライド・無料 |
| **「機械学習システムデザイン」** | Chip Huyen（著）、斉藤隆弘（訳） | 📖 | MLOps・本番 ML の実践バイブル。日本語訳あり（オライリー） |
| **「実践MLOps」** | 佐藤光紀ほか | 📖 | AWS を使った ML システム構築と運用。日本語（オーム社） |
| **「事例でわかるMLOps」** | 杉山阿聖, 太田満久, 久井裕貴 | 📖 | 技術・プロセス・文化の3面から学ぶ MLOps 入門（講談社） |
| **「仕事ではじめる機械学習」** | 有賀康顕, 中山心太, 西林孝 | 📖 | 日本語で読める実務 ML ガイド（オライリー）。現場のノウハウが豊富 |

---

## フェーズ 7: 実践プロジェクトと継続学習（12 ヶ月目〜）

### 7-1. ポートフォリオ作成
- Kaggle コンペへの参加
- GitHub へのプロジェクト公開
- 論文読み・実装（Papers With Code 活用）

### 7-2. 継続学習リソース
- **arXiv.org**: 最新論文
- **Hugging Face**: モデル・データセット・チュートリアル
- **fast.ai**: 実践的深層学習コース（無料）
- **Deep Learning Specialization（Coursera）**: Andrew Ng 講師

---

## 全体スケジュール概要

```
月  1 │ 数学基礎（線形代数・微積分）＋ Python 基礎
月  2 │ 統計基礎 ＋ NumPy / Pandas / Matplotlib
月  3 │ 教師あり学習（回帰・分類）＋ scikit-learn
月  4 │ 教師なし学習・モデル評価・特徴量エンジニアリング
月  5 │ 深層学習基礎（MLP・CNN）＋ PyTorch 入門
月  6 │ RNN / LSTM / Transformer・BERT / GPT
月  7 │ 専門分野選択①（NLP または CV）
月  8 │ 専門分野選択②（生成 AI・強化学習）
月  9 │ 応用 ML・アドバンスドトピック
月 10 │ MLOps・モデルデプロイ
月 11 │ Kaggle コンペ・実務プロジェクト
月 12 │ ポートフォリオ完成・就職活動・継続研究
```

---

## 無料で読める教材まとめ（🆓 のみ）

| フェーズ | タイトル | URL |
|---|---|---|
| 数学基礎 | Mathematics for Machine Learning（英語） | [mml-book.github.io](https://mml-book.github.io/) |
| 数学基礎 | メディカルAI専門コース 数学基礎（日本語） | [japan-medical-ai.github.io](https://japan-medical-ai.github.io/medical-ai-course-materials/notebooks/01_Basic_Math_for_ML.html) |
| 数学基礎 | 東工大「機械学習帳」（日本語） | [chokkan.github.io/mlnote](https://chokkan.github.io/mlnote/) |
| Python | Python 公式チュートリアル（日本語） | [docs.python.org/ja/3/tutorial](https://docs.python.org/ja/3/tutorial/) |
| Python / DL | 東大松尾研「DL4US」（日本語） | [github.com/matsuolab/DL4US](https://github.com/matsuolab/DL4US) |
| ML コア | PRML 日本語解説 PDF（普及版） | [herumi.github.io/prml](https://herumi.github.io/prml/) |
| ML コア | PRML 英語原書 PDF（Microsoft） | [Microsoft Research](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| ML コア | Interpretable ML 日本語訳 | [hacarus.github.io/interpretable-ml-book-ja](https://hacarus.github.io/interpretable-ml-book-ja/) |
| 深層学習 | Deep Learning book（英語） | [deeplearningbook.org](https://www.deeplearningbook.org/) |
| 深層学習 | 愛媛大学 深層学習講義 PDF（日本語） | [aiweb.cs.ehime-u.ac.jp](https://aiweb.cs.ehime-u.ac.jp/~ninomiya/enpitpro/deeplearning.pdf) |
| 深層学習 | ゼロから作るDL サンプルコード | [github.com/oreilly-japan/deep-learning-from-scratch](https://github.com/oreilly-japan/deep-learning-from-scratch) |
| NLP | キカガク NLP チュートリアル（日本語） | [free.kikagaku.ai/tutorial/basic_of_nlp](https://free.kikagaku.ai/tutorial/basic_of_nlp/) |
| NLP | Hugging Face NLP Course（日本語） | [huggingface.co/learn/nlp-course/ja](https://huggingface.co/learn/nlp-course/ja/chapter1/1) |
| MLOps | MLOps Yearning スライド（日本語） | [slideshare.net](https://www.slideshare.net/rakutentech/mlops-yearning) |

---

## 参考リンク

- [roadmap.sh/machine-learning](https://roadmap.sh/machine-learning)
- [roadmap.sh/ai-data-scientist](https://roadmap.sh/ai-data-scientist)
- [roadmap.sh/mlops](https://roadmap.sh/mlops)
- [github.com/mrdbourke/machine-learning-roadmap](https://github.com/mrdbourke/machine-learning-roadmap)
