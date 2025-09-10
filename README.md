# MNIST-Min: 最小可用的手寫數字辨識專案（PyTorch）

這是一個**入門且有趣**的手寫數字辨識模型（0–9），
使用 **PyTorch + torchvision** 自動下載 MNIST 資料集，
提供完整流程：**下載資料 → 訓練 → 評估 → 推論**。

## 特色
- 小而完整：單檔 `train.py` 就能完成訓練與測試。
- 模型簡單：一個小型 CNN，1–3 個 epoch 即可收斂到 98%+。
- 推論友善：`infer.py` 可對單張 PNG 進行預測，會輸出每個類別機率。

## 環境需求
- Python 3.9+
- 安裝依賴：
  ```bash
  pip install -r requirements.txt
  ```

> 若你在 macOS / Linux，且沒有 GPU，也可以用 CPU 版本的 PyTorch。

## 快速開始

### 1) 訓練與評估
```bash
python train.py --epochs 2 --batch-size 128
```
- 第一次會自動下載 MNIST（約 11MB）。
- 訓練結束會輸出：`model.pth` 與 `class_names.json`。

### 2) 產生範例圖片（可選）
```bash
python make_samples.py --count 10
```
- 會在 `samples/` 內存 10 張 `png` 和 `labels.csv`。

### 3) 單張圖片推論
```bash
python infer.py samples/0.png
```
- 會印出預測數字與機率分佈。

## 專案結構
```
mnist_min/
├─ train.py            # 訓練 + 測試 + 儲存模型
├─ infer.py            # 對單張 PNG 進行推論
├─ make_samples.py     # 從測試集匯出少量 PNG 樣本
├─ requirements.txt
├─ class_names.json    # 類別列表（訓練後生成）
├─ model.pth           # 已訓練模型（訓練後生成）
└─ samples/            # 範例 PNG 與 labels.csv（由 make_samples.py 生成）
```

## 小提示
- 想更快訓練：把 `--epochs` 改成 3 或 5。
- 想看更漂亮的曲線圖：可以自行加上 `matplotlib` 記錄 loss/accuracy。
- 想導出 ONNX：我可以幫你加一個 `export_onnx.py`。

---

祝你玩得開心！🎉 需要我做成 Jupyter Notebook 版或加上決策可視化，跟我說一聲就好。

---

## 進階功能
- **ONNX 匯出**：`python export_onnx.py --model model.pth --out model.onnx`
- **訓練曲線**：`python train.py --epochs 3 --save-plots` 或 `python plot_history.py`
- **Notebook**：開啟 `mnist_min.ipynb`，一鍵跑完整流程。
