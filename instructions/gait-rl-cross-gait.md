# タスク指示書：BipedalWalker-v3 の交互歩行獲得に向けた報酬ラッパー修正と再学習

## 1. 目的

前回の学習では「体幹の直立」は獲得できたが、「片脚が常に前、もう片脚が常に後ろに固定されたまま進む」という報酬ハッキングが発生した。
報酬設計を修正して「脚の前後固定」を禁止し、「後ろの脚を前に振り出して接地する（左右交互歩行）」を強く誘導して再学習を行う。

## 2. 環境前提

- OS: Linux (Dev Container / ヘッドレス環境)
- Python実行パス: **`/opt/venv/bin/python`**（必ずこのフルパスを使用すること）
- 導入済みライブラリ: `gymnasium[box2d]`, `stable-baselines3`, `tensorboard`, `imageio[ffmpeg]`
- 計算リソース: CPU実行（4並列 `SubprocVecEnv`）

## 3. 成果物

1. `train_alternating_gait.py`（修正版カスタム報酬ラッパーと学習・録画スクリプト）
2. `./videos_alternating/` 以下の段階別MP4動画（200k, 400k, 600k, 800k, 1000k ステップ時点）
3. `./models/ppo_bipedal_alternating_1000k.zip`（学習済みモデル）
4. `./tb_logs/` 以下の TensorBoard ログ（`PPO_AlternatingGait`）

## 4. 実装仕様（`train_alternating_gait.py`）

### 4-1. カスタム報酬ラッパー `AlternatingGaitRewardWrapper(gym.Wrapper)` の実装

以下のロジックを実装すること。

- **使用する観測値のインデックス:**
  - `obs[0]`: 体幹の傾き角度
  - `obs[1]`: 体幹の角速度
  - `obs[2]`: 前進速度（vel_x）
  - `obs[4]`: 脚1（左）の股関節角度
  - `obs[9]`: 脚2（右）の股関節角度
  - `obs[8]`: 脚1（左）の接地フラグ（`> 0.5` で接地）
  - `obs[13]`: 脚2（右）の接地フラグ（`> 0.5` で接地）

- **内部状態（`reset()` 時に初期化すること）:**
  - `self.prev_lead_leg = 0`（現在どちらの脚が前にあるか：`1`=脚1が前、`-1`=脚2が前、`0`=未確定）
  - `self.same_lead_steps = 0`（同じ脚が前に固定されている連続ステップ数）
  - `self.waiting_for_contact_leg = 0`（交差直後に接地を待っている脚：`1` または `2`、`0`=待機なし）
  - `self.leg1_contact_steps = 0`
  - `self.leg2_contact_steps = 0`

- **`step(self, action)` 内の報酬計算ロジック:**
  1. `obs, reward, terminated, truncated, info = self.env.step(action)` を実行する。
  2. **体幹の直立保持ペナルティ（前回同様・効果あり）:**
     - `posture_penalty = -0.3 * abs(obs[0]) - 0.05 * abs(obs[1])`
  3. **脚の入れ替え（交差）判定と「前後固定ペナルティ」:**
     - `hip_diff = obs[4] - obs[9]` とする。
     - `crossover_bonus = 0.0`、`stagnation_penalty = 0.0` とする。
     - `abs(hip_diff) > 0.15` のとき、`current_lead = 1 if hip_diff > 0 else -1` とする。
       - もし `self.prev_lead_leg != 0` かつ `current_lead != self.prev_lead_leg`（脚の前後が入れ替わった）場合：
         - 前進している（`obs[2] > 0.05`）なら `crossover_bonus = 1.0` を付与する。
         - `self.waiting_for_contact_leg = 1 if current_lead == 1 else 2` をセットする（前に出た脚の接地を待つ）。
         - `self.same_lead_steps = 0` にリセットする。
         - `self.prev_lead_leg = current_lead` を更新する。
       - もし `self.prev_lead_leg == 0` なら `self.prev_lead_leg = current_lead` とする。
       - 同じ脚が前のままの場合：`self.same_lead_steps += 1`。
     - **前後固定ペナルティ:** `self.same_lead_steps > 50` の場合、`stagnation_penalty = -0.15` を毎ステップ与える（片脚固定の歩容を強く抑制する）。
  4. **「振り出した脚の接地」ボーナス（真の交互一歩）:**
     - `leg1_contact = obs[8] > 0.5`、`leg2_contact = obs[13] > 0.5` とする。
     - `step_contact_bonus = 0.0` とする。
     - `self.waiting_for_contact_leg == 1 and leg1_contact`、または `self.waiting_for_contact_leg == 2 and leg2_contact` が成立した場合：
       - `step_contact_bonus = 2.0` を付与し、`self.waiting_for_contact_leg = 0` にリセットする。
  5. **引きずりペナルティ:**
     - 各脚の連続接地ステップ数をカウントし、どちらかが `> 30` ステップ連続で接地している場合は `drag_penalty = -0.1` を与える（それ以外は `0.0`）。
  6. **最終報酬:**
     - `custom_reward = reward + posture_penalty + crossover_bonus + step_contact_bonus + stagnation_penalty + drag_penalty`
     - `return obs, custom_reward, terminated, truncated, info`

### 4-2. 学習および録画の仕様（`if __name__ == "__main__":` 内）

- **学習環境:**
  - `SubprocVecEnv([lambda: AlternatingGaitRewardWrapper(gym.make("BipedalWalker-v3")) for _ in range(4)])` を使用する。
- **PPOハイパーパラメータ:**
  - `PPO("MlpPolicy", train_env, verbose=1, ent_coef=0.015, tensorboard_log="./tb_logs/")`
- **学習ループ:**
  - 合計 **1,000,000ステップ**（200,000ステップ × 5回）学習する。
  - 各ループで `model.learn(total_timesteps=200_000, reset_num_timesteps=False, tb_log_name="PPO_AlternatingGait")` を実行する。
  - 各200,000ステップ完了ごとに、`./videos_alternating/step_{total // 1000}k.mp4` として評価動画を保存する（`render_mode="rgb_array"`, 最大1600ステップ, `imageio.mimsave(..., fps=30)`）。
- **モデル保存:**
  - 学習完了後に `./models/ppo_bipedal_alternating_1000k` に保存し、`train_env.close()` を呼ぶ。

## 5. 実行および検証手順

1. `train_alternating_gait.py` を作成する。
2. `/opt/venv/bin/python train_alternating_gait.py` を実行する。
3. 実行完了後、`./videos_alternating/` 内に5本の `.mp4` ファイルと `./models/ppo_bipedal_alternating_1000k.zip` が生成されていることを確認する。
4. 完了報告として、学習前後の `ep_rew_mean` の推移と生成された動画のパス一覧を出力する。

## 6. 制約・注意事項

- Pythonコマンドは必ず `/opt/venv/bin/python` を使用すること。
- 追加の `pip install` や `apt-get` は行わないこと。
- `tensorboard` は起動しないこと。
