# タスク指示書：SAC学習済みモデルを用いた BipedalWalker-v3 の片麻痺（片脚筋力制限）適応学習

## 1. 目的

正常歩行を習得済みの SAC モデル（`./models/sac_bipedal_500k.zip`）をベースに、片脚（脚2）の関節トルクを30%に制限する「簡易的な片麻痺環境」を構築する。
筋力制限をかけた直後（0kステップ）の歩容と、そこから追加学習（ファインチューニング）を行って代償歩行を獲得していく過程を MP4 動画として記録・比較する。

## 2. 環境前提

- OS: Linux (Dev Container / ヘッドレス環境)
- Python実行パス: **`/opt/venv/bin/python`**（必ずこのフルパスを使用すること）
- 既存の学習済みモデル: `./models/sac_bipedal_500k.zip`
- 導入済みライブラリ: `gymnasium[box2d]`, `stable-baselines3`, `tensorboard`, `imageio[ffmpeg]`

## 3. 成果物

1. `train_sac_hemiplegia.py`（片脚筋力制限ラッパーと追加学習・録画スクリプト）
2. `./videos_hemiplegia/` 以下の段階別 MP4 動画（`step_000k.mp4`, `step_100k.mp4`, `step_200k.mp4`, `step_300k.mp4`）
3. `./models/sac_bipedal_hemiplegia_300k.zip`（適応学習後のモデル）
4. `./tb_logs/` 以下の TensorBoard ログ（`SAC_Hemiplegia`）

## 4. 実装仕様（`train_sac_hemiplegia.py`）

### 4-1. 片脚筋力制限ラッパー `HemiplegiaWrapper(gym.Wrapper)` の実装

`BipedalWalker-v3` の行動空間（4次元：`[脚1股関節, 脚1膝, 脚2股関節, 脚2膝]`）に対して、片脚の最大発揮トルクを制限するラッパーを実装すること。

- `__init__(self, env, strength_ratio=0.3)`:
  - `self.strength_ratio = strength_ratio` を保持する（デフォルトは 0.3＝正常の30%の筋力）。
- `step(self, action)`:
  - 元の配列を破壊しないよう `modified_action = np.array(action, dtype=np.float32, copy=True)` でコピーを作成する。
  - **脚2（患側）の筋力制限:** `modified_action[2] *= self.strength_ratio` および `modified_action[3] *= self.strength_ratio` を適用する。
  - `obs, reward, terminated, truncated, info = self.env.step(modified_action)` を実行する。
  - **転倒回避（生存）の意識づけ:** 転倒していない（`not terminated`）場合は `reward += 0.05` を加算する。
  - `return obs, reward, terminated, truncated, info` を返す。

### 4-2. 動画録画関数 `record_video(env_id, model, filename, strength_ratio=0.3, steps=1600)` の実装

- **重要:** 評価時も必ず `HemiplegiaWrapper(gym.make(env_id, render_mode="rgb_array"), strength_ratio=strength_ratio)` でラップした環境を使用すること。
- `model.predict(obs, deterministic=True)` で行動を選択し、最大 1600 ステップ実行する。
- `imageio.mimsave(filename, frames, fps=30)` で MP4 動画として保存し、`env.close()` を呼ぶ。

### 4-3. メイン処理（`if __name__ == "__main__":` 内）

- `./videos_hemiplegia`, `./models`, `./tb_logs` ディレクトリを作成する。
- 学習環境を `train_env = HemiplegiaWrapper(gym.make("BipedalWalker-v3"), strength_ratio=0.3)` で作成する。
- 既存の正常歩行モデルを読み込む：
  ```python
  model = SAC.load(
      "./models/sac_bipedal_500k",
      env=train_env,
      tensorboard_log="./tb_logs/",
  )
  ```
- **発症直後（追加学習前：0k）の録画:**
  - 学習を開始する前に、`record_video("BipedalWalker-v3", model, "./videos_hemiplegia/step_000k.mp4", strength_ratio=0.3)` を実行し、「正常歩行の脳のまま突然片脚が動かなくなった直後」の様子を録画する。
- **適応学習（ファインチューニング）ループ:**
  - 合計 **300,000 ステップ**（100,000 ステップ × 3回）の追加学習を行う。
  - 各ループで `model.learn(total_timesteps=100_000, reset_num_timesteps=False, tb_log_name="SAC_Hemiplegia")` を実行する。
  - 各 100,000 ステップ完了ごとに、`./videos_hemiplegia/step_{100k, 200k, 300k}.mp4` を録画する。
- **モデル保存:**
  - 学習完了後に `model.save("./models/sac_bipedal_hemiplegia_300k")` で保存し、`train_env.close()` を呼ぶ。

## 5. 実行および検証手順

1. `train_sac_hemiplegia.py` を作成する。
2. `/opt/venv/bin/python train_sac_hemiplegia.py` を実行する。
3. 実行完了後、以下が生成されていることをシェルコマンドで確認する：
   - `./videos_hemiplegia/` 内に 4本（`step_000k.mp4` 〜 `step_300k.mp4`）の `.mp4` ファイルがあること。
   - `./models/sac_bipedal_hemiplegia_300k.zip` があること。
4. 完了報告として、学習前後の `ep_rew_mean` の推移と、生成された動画ファイルのパス一覧を出力する。

## 6. 制約・注意事項

- Pythonコマンドは必ず `/opt/venv/bin/python` を使用すること。
- 追加の `pip install` や `apt-get` は行わないこと。
- `tensorboard` は起動しないこと。
- `./models/sac_bipedal_500k.zip` が見つからない場合は、エラーを出してユーザーにパスを確認すること。
