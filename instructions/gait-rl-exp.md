# タスク指示書：BipedalWalker-v3 の強化学習パイプライン構築と実行

## 1. 目的

Dev Container 環境内で `BipedalWalker-v3` の二足歩行を強化学習（PPO）で学習させ、学習過程の歩容の変化を段階的なMP4動画およびTensorBoardログとして記録・出力する。

## 2. 環境前提

- OS: Linux (Dev Container / ヘッドレス環境・GUIディスプレイなし)
- Python: `/opt/venv` の仮想環境が有効化済み
- 導入済みライブラリ: `gymnasium[box2d]`, `stable-baselines3`, `tensorboard`, `moviepy`, `imageio[ffmpeg]`
- 計算リソース: CPU実行（マルチコア並列化を利用）

## 3. 成果物（作成・生成するもの）

1. `train_bipedal.py`（学習・段階録画・モデル保存を行うPythonスクリプト）
2. `./videos/` 以下の段階別MP4動画（0k, 100k, 200k, 300k, 400k, 500k ステップ時点）
3. `./models/ppo_bipedal_500k.zip`（学習済みモデル）
4. `./tb_logs/` 以下の TensorBoard 学習ログ

## 4. 実装仕様（`train_bipedal.py`）

以下の要件をすべて満たすスクリプトをワークスペースのルートに作成すること。

- **動画録画関数 `record_video(model, step_label, video_dir="./videos")` の実装:**
  - `gym.make("BipedalWalker-v3", render_mode="rgb_array")` を使用する（ヘッドレス環境のため `human` は絶対に使わないこと）。
  - `gymnasium.wrappers.RecordVideo` でラップし、`name_prefix=f"step_{step_label}"`、`episode_trigger=lambda x: x == 0` を設定する。
  - 最大1600ステップ実行し、`model is None` の場合は `env.action_space.sample()`、それ以外は `model.predict(obs, deterministic=True)` で行動を選択する。
  - `terminated or truncated` でループを抜け、必ず `env.close()` を呼ぶ。
- **メイン処理（`if __name__ == "__main__":` ブロック内に記述すること）:**
  - `./videos` と `./models` ディレクトリを作成する（`os.makedirs(..., exist_ok=True)`）。
  - 学習前（`model=None`, `step_label="000k"`）の動画を1本録画する。
  - `stable_baselines3.common.env_util.make_vec_env` と `SubprocVecEnv` を使い、`BipedalWalker-v3` を **4並列（`n_envs=4`）** で作成する。
  - `PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./tb_logs/")` でモデルを初期化する。
  - **100,000ステップごとに5回（合計500,000ステップ）** ループで学習を回す。
    - `model.learn(total_timesteps=100_000, reset_num_timesteps=False, tb_log_name="PPO_Bipedal")` を呼び出す（タイムステップをリセットしないこと）。
    - 各100,000ステップ完了ごとに `record_video(model=model, step_label=f"{total // 1000}k")` で動画を保存する。
  - 最後に `model.save("./models/ppo_bipedal_500k")` でモデルを保存し、`train_env.close()` を呼ぶ。

## 5. 実行および検証手順

1. `train_bipedal.py` を作成する。
2. まず構文エラーやインポートエラーがないか確認した上で、`python train_bipedal.py` を実行する。
3. 実行完了後、以下のすべてが存在することをシェルコマンド（`ls -la` 等）で確認する：
   - `./videos/` 内に 6本（`step_000k` 〜 `step_500k`）の `.mp4` ファイルが生成されていること。
   - `./models/ppo_bipedal_500k.zip` が生成されていること。
   - `./tb_logs/` 内にログディレクトリが生成されていること。
4. 完了報告として、以下を簡潔にまとめて出力する：
   - 学習前後の `ep_rew_mean`（平均報酬）の推移
   - 生成された動画ファイルのパス一覧
   - ユーザーが後からTensorBoardを起動するためのコマンド（`tensorboard --logdir ./tb_logs`）

## 6. 制約・注意事項

- 追加の `pip install` や `apt-get` は行わないこと（既存環境で動作する）。
- `tensorboard --logdir ./tb_logs` はフォアグラウンドでプロセスをブロックするため、エージェント自身では起動せず、完了報告でユーザーにコマンドを案内するにとどめること。
- エラーが発生した場合は、エラーログの原因を特定し、`train_bipedal.py` を修正して再実行すること。
