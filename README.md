# 手話認識 (Isolated Sign Language Recognition)

東大の授業課題：WLASL100 データセットを用いた Transformer ベースの孤立手話認識。

## 構成
- `main.py` — ローカル実行用の学習・評価スクリプト（Single Split, Golden Config）
- `colab_script.py` — Google Colab 用スクリプト
- `model_optimized.py` / `model_signjoey_inspired.py` — モデルのバリエーション
- `result_temp/` — 実験結果（チェックポイント・学習履歴）
- `presentation/` — 期末発表資料（PDF / PPTX）

## 結果
Transformer (dim=128, layers=2, dropout=0.2, 200 epochs): **Test Accuracy 77.13%**

データセット (`dataset/wlasl100`) と授業提供のチュートリアルモジュール (`slt_tutorials/`, © N. Takayama) は著作権の都合によりリポジトリに含まれていません。実行するには別途 `slt_tutorials/modules/` を配置してください。
