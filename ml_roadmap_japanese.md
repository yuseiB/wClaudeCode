# 機械学習 学習ロードマップ（roadmap.sh 準拠）

> **参考元**: [roadmap.sh/machine-learning](https://roadmap.sh/machine-learning) / [roadmap.sh/ai-data-scientist](https://roadmap.sh/ai-data-scientist)
> **対象期間**: 約 9〜12 ヶ月（プログラミング経験者の場合 6 ヶ月も可能）

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

### 推薦教科書
| タイトル | 著者 | 特徴 |
|---|---|---|
| **「機械学習のための数学」** (*Mathematics for Machine Learning*) | Deisenroth, Faisal, Ong | ML に必要な数学を網羅、無料 PDF あり |
| **「統計学入門」** | 東京大学教養学部統計学教室 | 日本語で読める定番統計入門 |
| **「線形代数入門」** | 松坂和夫 | 日本語の丁寧な線形代数テキスト |

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

### 推薦教科書
| タイトル | 著者 | 特徴 |
|---|---|---|
| **「Pythonチュートリアル」** | Guido van Rossum（公式ドキュメント） | 公式・無料・日本語あり |
| **「Pythonによるデータ分析入門」** | Wes McKinney（著）、小林儀匡（訳） | Pandas 作者による実践書 |
| **「Python機械学習プログラミング」** | Sebastian Raschka, Vahid Mirjalili | scikit-learn を使った実践入門（日本語訳あり） |

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

### 推薦教科書
| タイトル | 著者 | 特徴 |
|---|---|---|
| **「scikit-learnとTensorFlowによる実践機械学習」** | Aurélien Géron（著）、下田倫大（訳） | 最も有名な ML 実践書。第3版まで日本語訳あり |
| **「機械学習 理論から実践へ」** (*Understanding Machine Learning*) | Shalev-Shwartz, Ben-David | 理論的背景をしっかり学びたい人向け |
| **「パターン認識と機械学習」（PRML）** | Christopher M. Bishop（著）、元田浩ほか（訳） | ML の聖典。上下2巻で日本語訳あり |

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

### 推薦教科書
| タイトル | 著者 | 特徴 |
|---|---|---|
| **「深層学習」（Deep Learning）** | Goodfellow, Bengio, Courville（著）、岩澤有祐ほか（訳） | 深層学習の教科書。日本語訳あり（通称 GBC 本） |
| **「ゼロから作るDeep Learning」シリーズ** | 斎藤康毅 | Python で一から実装。日本語オリジナル。大人気シリーズ |
| **「PyTorchによる自然言語処理」** | Delip Rao, Brian McMahan（著）、和田崇仁（訳） | PyTorch と NLP の実践書 |

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

### 推薦教科書
| タイトル | 著者 | 特徴 |
|---|---|---|
| **「自然言語処理の基礎」** | 岡崎直観, 荒瀬由紀, 鈴木潤ほか | 日本語 NLP の標準テキスト |
| **「Transformerによる自然言語処理」** | Lewis Tunstall ほか（著）、中山光樹（訳） | Hugging Face 公式著者による実践書 |
| **「コンピュータビジョン最前線」** | 井上中順ほか | 日本語 CV 解説書シリーズ |

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

### 推薦教科書
| タイトル | 著者 | 特徴 |
|---|---|---|
| **「機械学習システムデザイン」** | Chip Huyen（著）、斉藤隆弘（訳） | MLOps・本番 ML の実践バイブル（日本語訳あり） |
| **「仕事ではじめる機械学習」** | 有賀康顕, 中山心太, 西林孝 | 日本語で読める実務 ML ガイド |

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

## 優先度別 推薦教科書まとめ

### ★★★ 必読（入門〜中級）
1. **「ゼロから作るDeep Learning」** — 斎藤康毅（日本語オリジナル、実装力強化に最適）
2. **「scikit-learnとTensorFlowによる実践機械学習」** — Géron 著（日本語訳あり）
3. **「機械学習のための数学」** — Deisenroth ほか（無料 PDF、英語）

### ★★★ 必読（中級〜上級）
4. **「深層学習」(GBC)** — Goodfellow ほか（日本語訳あり、理論の深掘りに）
5. **「パターン認識と機械学習」(PRML)** — Bishop（日本語訳あり、ML の聖典）
6. **「機械学習システムデザイン」** — Chip Huyen（日本語訳あり、実務向け）

### ★★ 推薦（専門分野）
7. **「自然言語処理の基礎」** — 岡崎直観ほか（NLP 日本語標準テキスト）
8. **「Transformerによる自然言語処理」** — Tunstall ほか（Hugging Face 実践）
9. **「仕事ではじめる機械学習」** — 有賀ほか（実務 ML 日本語ガイド）

---

## 参考リンク

- [roadmap.sh/machine-learning](https://roadmap.sh/machine-learning)
- [roadmap.sh/ai-data-scientist](https://roadmap.sh/ai-data-scientist)
- [roadmap.sh/mlops](https://roadmap.sh/mlops)
- [github.com/mrdbourke/machine-learning-roadmap](https://github.com/mrdbourke/machine-learning-roadmap)
