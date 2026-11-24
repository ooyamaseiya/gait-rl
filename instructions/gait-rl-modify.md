# タスク指示書：BipedalWalker-v3 の正常歩行（左右交互歩行）に向けた報酬設計と再学習

## 1. 目的

前回の学習で発生した「片脚引きずり（左右非対称・交互性の欠如・体幹の後傾）」を改善するため、カスタム報酬ラッパー（`NormalGaitRewardWrapper`）を導入して PPO で再学習を行い、左右交互に脚を振り出す正常歩行に近い歩容を獲得する。

## 2. 環境前提

- OS: Linux (Dev Container / ヘッドレス環境)
- Python: `/opt/venv` が有効化済み
- 導入済みライブラリ: `gymnasium[box2d]`, `stable-baselines3`, `tensorboard`, `moviepy`, `imageio[ffmpeg]`
- 計算リソース: CPU実行（4並列 `SubprocVecEnv`）

## 3. 成果物

1. `train_normal_gait.py`（カスタム報酬ラッパーと学習・録画パイプラインを実装したスクリプト）
2. `./videos_normal/` 以下の段階別MP4動画（200k, 400k, 600k, 800k, 1000k ステップ時点）
3. `./models/ppo_bipedal_normal_1000k.zip`（学習済みモデル）
4. `./tb_logs/` 以下の TensorBoard ログ（`PPO_NormalGait`）

## 4. 実装仕様（`train_normal_gait.py`）

### 4-1. カスタム報酬ラッパー `NormalGaitRewardWrapper(gym.Wrapper)` の実装

`BipedalWalker-v3` の観測値 `obs` を用いて、以下の4つの要素を元の報酬 `reward` に加算・減算するラッパーを実装すること。

- **使用する観測値のインデックス:**
  - `obs[0]`: 体幹の傾き角度（hull angle, 0が直立）
  - `obs[1]`: 体幹の角速度（hull angular velocity）
  - `obs[4]`: 脚1（左）の股関節角度
  - `obs[9]`: 脚2（右）の股関節角度
  - `obs[8]`: 脚1（左）の接地フラグ（0.0 または 1.0）
  - `obs[13]`: 脚2（右）の接地フラグ（0.0 または 1.0）

- **内部状態（`reset()` 時に初期化すること）:**
  - `self.last_contact_leg`: 最後に単脚接地した脚（`None`, `1`, `2`）
  - `self.prev_hip_diff_sign`: 前ステップの `obs[4] - obs[9]` の符号（`0`, `1`, `-1`）
  - `self.leg1_contact_steps`: 脚1の連続接地ステップ数（初期値 `0`）
  - `self.leg2_contact_steps`: 脚2の連続接地ステップ数（初期値 `0`）

- **`step(self, action)` 内の報酬計算ロジック:**
  1. `obs, reward, terminated, truncated, info = self.env.step(action)` を実行する。
  2. **体幹の直立保持ペナルティ（後傾・揺れの抑制）:**
     - `posture_penalty = -0.3 * abs(obs[0]) - 0.05 * abs(obs[1])`
  3. **左右交互の接地ボーナス（片脚固定の解消）:**
     - `leg1_contact = obs[8] > 0.5`、`leg2_contact = obs[13] > 0.5` とする。
     - `alternating_bonus = 0.0` とし、単脚接地（片方のみ `True`）かつ `self.last_contact_leg` と逆の脚が接地した場合に `alternating_bonus = 0.5` を付与し、`self.last_contact_leg` を更新する。
  4. **股関節の前後交差ボーナス（脚の入れ替え誘導）:**
     - `hip_diff = obs[4] - obs[9]` とし、`|hip_diff| > 0.1` のときに現在の符号（`1` or `-1`）を判定する。
     - `self.prev_hip_diff_sign` が `0` でなく、かつ符号が反転した（後ろの脚が前に出た）場合に `crossover_bonus = 0.5` を付与し、`self.prev_hip_diff_sign` を更新する（それ以外は `0.0`）。
  5. **片脚引きずりペナルティ（長時間の連続接地を抑制）:**
     - 接地中は各脚の連続接地カウントを `+1`、離地したら `0` にリセットする。
     - `drag_penalty = 0.0` とし、`self.leg1_contact_steps > 35` または `self.leg2_contact_steps > 35` の場合に `drag_penalty -= 0.1` を与える。
  6. **最終報酬:**
     - `custom_reward = reward + posture_penalty + alternating_bonus + crossover_bonus + drag_penalty`
     - `return obs, custom_reward, terminated, truncated, info`

### 4-2. 学習および録画の仕様（`if __name__ == "__main__":` 内）

- **学習環境:**
  - `make_vec_env("BipedalWalker-v3", n_envs=4, vec_env_cls=SubprocVecEnv, wrapper_class=NormalGaitRewardWrapper)` を使用する。
- **PPOハイパーパラメータ（探索を広げて局所最適解を回避する）:**
  - `PPO("MlpPolicy", train_env, verbose=1, ent_coef=0.01, tensorboard_log="./tb_logs/")`
- **学習ループ:**
  - 合計 **1,000,000ステップ**（200,000ステップ × 5回）学習する。
  - 各ループで `model.learn(total_timesteps=200_000, reset_num_timesteps=False, tb_log_name="PPO_NormalGait")` を実行する。
  - 各200,000ステップ完了ごとに、`record_video(model, step_label=f"{total // 1000}k", video_dir="./videos_normal")` で動画を録画する（録画時は `render_mode="rgb_array"` を使用し、最大1600ステップ）。
- **モデル保存:**
  - 学習完了後に `model.save("./models/ppo_bipedal_normal_1000k")` で保存し、`train_env.close()` を呼ぶ。

## 5. 実行および検証手順

1. `train_normal_gait.py` を作成する。
2. `python train_normal_gait.py` を実行する。
3. 実行完了後、以下が生成されていることをシェルコマンドで確認する：
   - `./videos_normal/` 内に 5本（`step_200k` 〜 `step_1000k`）の `.mp4` ファイルがあること。
   - `./models/ppo_bipedal_normal_1000k.zip` があること。
4. 完了報告として、学習前後の `ep_rew_mean` の推移と、生成された動画ファイルのパス一覧を出力する。

## 6. 制約・注意事項

- 追加の `pip install` や `apt-get` は行わないこと。
- `tensorboard` はフォアグラウンドをブロックするため起動しないこと。
- エラーが発生した場合は原因を修正して再実行すること。
