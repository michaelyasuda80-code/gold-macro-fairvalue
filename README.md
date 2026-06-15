# Gold Macro Fair Value

ドル建て金（COMEX 期近先物 `GC=F`）のマクロ要因による理論価格を推定し、
**実勢価格との乖離 (statistical arbitrage)** をダッシュボードで可視化します。

- **データソース**: Yahoo Finance のみ（API キー不要、完全無料）
- **モデル**: log(Gold) を実質金利プロキシ・DXY・原油・VIX・S&P・BTC・USDCNY 等で説明する OLS
- **シグナル**: 残差の z-score (|z|>2 で割安/割高)
- **寄与度**: 各ファクターが現在の理論価格を何 % 押し上げ/押し下げているか

## ローカル実行

```powershell
# 1. Python 3.11 か 3.12 を Microsoft Store または python.org からインストール
# 2. 仮想環境を作る
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. 依存をインストール
pip install -r requirements.txt

# 4. アプリ起動
streamlit run app.py
```

ブラウザで `http://localhost:8501` が開きます。

## デプロイ — Phase 1: Streamlit Community Cloud（今すぐ）

1. このディレクトリを **public** な GitHub リポジトリに push
   ```powershell
   git init
   git add .
   git commit -m "initial commit"
   git branch -M main
   git remote add origin https://github.com/<your-handle>/goldprice.git
   git push -u origin main
   ```
2. https://share.streamlit.io にログイン（GitHub アカウント）
3. **New app** → リポジトリ・ブランチ `main`・Main file `app.py` を選択 → **Deploy**
4. 数分で `https://<your-handle>-goldprice.streamlit.app` のような URL が発行されます
5. その URL を X でシェア

スリープ運用なので初回アクセスが 30 秒ほど遅いことがあります。
頻繁にアクセスがあれば常時起動状態になります。

## デプロイ — Phase 2: GitHub Pages（後で・恒久URL）

Streamlit Cloud から脱却して、GitHub Pages の静的サイトに移行する場合:

1. `build_site.py`（別途用意）で Plotly 図を `docs/index.html` に書き出す
2. `.github/workflows/build.yml` で毎朝 cron 実行 → docs/ に commit
3. Repo Settings → Pages → Source = `main / docs` で公開
4. カスタムドメインも紐付け可

このフェーズに移ると、Streamlit のスリープ問題が消え、URL も
`https://<your-handle>.github.io/goldprice/` で固定。X 投稿用のリンクとして最適。

## モデルの読み方

- **R²** が 0.85 を超えていれば、選んだファクターで金価格の動きの大半を説明できている
- **z-score が +2 以上** → 金は理論値より高い → マクロから見ると割高（売り側に寄る材料）
- **z-score が −2 以下** → 金は理論値より低い → 割安（買い側に寄る材料）
- **寄与度の棒グラフ** で「今日の理論値は何によって押し上げられているのか」が一目で分かる

例: 「実質金利の寄与が −15%、DXY が −5%、原油が +3% で、合計の理論値は基準より
−17%」のように読みます。

## 注意

- 単純 OLS の残差は**自己相関**を持ちます。短期トレードシグナルとして使う場合は
  半減期 (half-life) を測って保有期間を決めてください
- Yahoo の期近先物は**ロール日**にギャップが出ることがあります
- 投資助言ではありません。リサーチ・教育目的での利用に限ります
