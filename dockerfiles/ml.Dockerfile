# Python向けのベースイメージ（必要に応じてバージョンを調整してください）
FROM mcr.microsoft.com/devcontainers/python:3.12-trixie

ARG PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
ARG GITLEAKS_VERSION=8.30.0
ARG GH_VERSION=2.62.0

# ユーザーに依存しない共有パスに配置する
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# MuJoCo / MyoSuite のオフスクリーン（画面なし）描画用環境変数
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# 1. システムパッケージ（音声処理・Box2Dビルド・MuJoCo描画）と uv のインストール
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        build-essential \
        swig \
        python3-dev \
        xvfb \
        patchelf \
        libgl1 \
        libglx-mesa0 \
        libosmesa6-dev \
        libglfw3 \
        libglew-dev \
        libglib2.0-0 \
    && pip install --no-cache-dir uv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

    # 2. Python 仮想環境の作成と機械学習・強化学習・音声処理パッケージのインストール
RUN python3 -m venv ${VIRTUAL_ENV} \
    && uv pip install --upgrade pip setuptools wheel \
    && uv pip install --no-cache \
        torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL} \
    && uv pip install --no-cache \
        "gymnasium[box2d]" \
        "myosuite>=3.0" \
        stable-baselines3 \
        tensorboard \
        matplotlib \
        "imageio[ffmpeg]" \
        moviepy \
        librosa \
        soundfile \
        onnx \
        onnxruntime \
        scikit-learn \
        pytest \
    && chmod -R a+rwX ${VIRTUAL_ENV}

# 3. opencode のインストール（全ユーザーから使えるよう /usr/local/bin へ配置）
RUN curl -fsSL https://opencode.ai/v2/install | bash \
    && mv /root/.opencode/bin/opencode /usr/local/bin/opencode \
    && rm -rf /root/.opencode

# 4. gitleaks と gh (GitHub CLI) のインストール（全ユーザーから使えるよう /usr/local/bin へ配置）
RUN ARCH=$(dpkg --print-architecture) \
    && case "${ARCH}" in \
         amd64) GITLEAKS_ARCH="x64"; GH_ARCH="amd64" ;; \
         arm64) GITLEAKS_ARCH="arm64"; GH_ARCH="arm64" ;; \
         *) echo "Unsupported architecture: ${ARCH}" && exit 1 ;; \
       esac \
    && curl -fsSL "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_${GITLEAKS_ARCH}.tar.gz" \
       -o /tmp/gitleaks.tar.gz \
    && tar -xzf /tmp/gitleaks.tar.gz -C /usr/local/bin gitleaks \
    && chmod 755 /usr/local/bin/gitleaks \
    && rm -f /tmp/gitleaks.tar.gz \
    && curl -fsSL "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_${GH_ARCH}.tar.gz" \
       -o /tmp/gh.tar.gz \
    && tar -xzf /tmp/gh.tar.gz -C /tmp \
    && mv /tmp/gh_${GH_VERSION}_linux_${GH_ARCH}/bin/gh /usr/local/bin/gh \
    && chmod 755 /usr/local/bin/gh \
    && rm -rf /tmp/gh.tar.gz /tmp/gh_${GH_VERSION}_linux_${GH_ARCH}
