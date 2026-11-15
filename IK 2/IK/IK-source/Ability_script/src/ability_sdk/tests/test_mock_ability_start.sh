#!/bin/bash
set -euo pipefail

# 获取能力工作目录
ABILITY_PACK_DIR="${ABILITY_FRAMEWORK_HOME}/packages/robot.test.org/0.1.0"
if [ ! -d "$ABILITY_PACK_DIR" ]; then
    echo "[BOOT] Error: Ability directory not found: $ABILITY_PACK_DIR" >&2
    exit 1
fi

# 1. 从package.yaml读取配置
PACKAGE_YAML="${ABILITY_PACK_DIR}/package.yaml"
if [ ! -f "$PACKAGE_YAML" ]; then
    echo "[BOOT] Error: package.yaml not found in $ABILITY_PACK_DIR" >&2
    exit 1
fi

# 解析package.yaml
PYTHON_VERSION=$(grep "requires-python" "$PACKAGE_YAML" | awk '{print $2}' | tr -d '"')
# INSTALL_MODE=$(grep "installMode" "$PACKAGE_YAML" | awk '{print $2}')

if [ -z "$PYTHON_VERSION" ]; then
    echo "[BOOT] Error: requires-python not specified in package.yaml" >&2
    exit 1
fi

# echo "[BOOT] Python version required: $PYTHON_VERSION"

# 2. 安装Python
if ! command -v uv >/dev/null 2>&1; then
    echo "[BOOT] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo "Error: Failed to install uv" >&2
        exit 1
    fi
fi

# 获取或安装指定版本的Python
PYTHON_PATH=$(uv python find "$PYTHON_VERSION" || true)
if [ -z "$PYTHON_PATH" ]; then
    echo "[BOOT] Python $PYTHON_VERSION not found, installing..."
    uv python install "$PYTHON_VERSION"
    PYTHON_PATH=$(uv python find "$PYTHON_VERSION")
    if [ -z "$PYTHON_PATH" ]; then
        echo "[BOOT] Error: Failed to install Python $PYTHON_VERSION" >&2
        exit 1
    fi
fi

# 获取Python的实际路径（解析可能的软链接）
REAL_PYTHON_PATH=$(readlink -f "$PYTHON_PATH" || echo "$PYTHON_PATH")
# echo "[BOOT] Using Python at: $REAL_PYTHON_PATH"

# 3. 配置环境
REQUIREMENTS_FILE="${ABILITY_PACK_DIR}/requirements.txt"
ENV_DIR="${ABILITY_PACK_DIR}/env"

if [ ! -d "$ENV_DIR" ]; then
    # 在线模式 - 使用requirements.txt
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        echo "[BOOT] Error: requirements.txt not found in online mode" >&2
        exit 1
    fi
    
    # echo "[BOOT] Creating virtual environment..."
    "$REAL_PYTHON_PATH" -m venv "$ENV_DIR"
    
    # echo "[BOOT] Installing dependencies from requirements.txt..."
    "${ENV_DIR}/bin/pip" install --disable-pip-version-check -r "$REQUIREMENTS_FILE"
    
    # 设置使用的Python路径为虚拟环境的Python
    PYTHON_TO_USE="${ENV_DIR}/bin/python"
    
else
    # 离线模式 - 使用env文件夹
    ENV_PYTHON_PATH="${ENV_DIR}/bin/python"
    if [ ! -f "$ENV_PYTHON_PATH" ]; then
        echo "[BOOT] Error: Python not found in env directory" >&2
        exit 1
    fi
    
    # 备份原始Python软链接
    if [ -L "$ENV_PYTHON_PATH" ]; then
        # echo "Backing up original python link..."
        mv "$ENV_PYTHON_PATH" "${ENV_PYTHON_PATH}.bak"
    fi
    
    # 更新env中的Python软链接
    # echo "[BOOT] Updating python symlink in env to use system Python..."
    ln -sf "$REAL_PYTHON_PATH" "$ENV_PYTHON_PATH"
    
    # 验证Python版本
    ENV_PYTHON_VERSION=$("$ENV_PYTHON_PATH" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    REQUIRED_MIN_VERSION=$(echo "$PYTHON_VERSION" | sed 's/[>=<]//g')
    
    if [ "$(printf '%s\n' "$REQUIRED_MIN_VERSION" "$ENV_PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_MIN_VERSION" ]; then
        echo "[BOOT] Error: Python version in env ($ENV_PYTHON_VERSION) does not meet requirement ($PYTHON_VERSION)" >&2
        exit 1
    fi

    # 激活虚拟环境
    source "${ENV_DIR}/bin/activate"
    
    PYTHON_TO_USE="$ENV_PYTHON_PATH"
fi

# 4. 启动能力脚本

echo "[BOOT] Python=$PYTHON_TO_USE, EnvDir=$ENV_DIR"

ABILITY_SCRIPT="/home/zpf/InsightOS/ability-sdk-python/tests/test_mock_ability.py"
if [ ! -f "$ABILITY_SCRIPT" ]; then
    echo "Error: Ability script not found: $ABILITY_SCRIPT" >&2
    exit 1
fi
$PYTHON_TO_USE "$ABILITY_SCRIPT" $1 $2 