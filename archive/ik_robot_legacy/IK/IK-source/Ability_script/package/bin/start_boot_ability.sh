#!/usr/bin/bash
set -euo pipefail

# 检查ABILITY_FRAMEWORK_HOME、PACKAGE_NAME、PACKAGE_VERSION、SCRIPT_DIR环境变量
if [ -z "${ABILITY_FRAMEWORK_HOME:-}" ]; then
    ABILITY_FRAMEWORK_HOME=$PWD
fi
if [ -z "${PACKAGE_NAME:-}" ]; then
    echo -e "\033[31m[BOOT] Error: PACKAGE_NAME is not set\033[0m" >&2
    exit 1
fi
if [ -z "${PACKAGE_VERSION:-}" ]; then
    echo -e "\033[31m[BOOT] Error: PACKAGE_VERSION is not set\033[0m" >&2
    exit 1
fi
if [ -z "${ABILITY_SCRIPT_ENTRY:-}" ]; then
    echo -e "\033[31m[BOOT] Error: ABILITY_SCRIPT_ENTRY is not set\033[0m" >&2
    exit 1
fi

# 获取能力工作目录
export ABILITY_PACK_DIR="${ABILITY_FRAMEWORK_HOME}/packages/${PACKAGE_NAME}/${PACKAGE_VERSION}"
if [ ! -d "$ABILITY_PACK_DIR" ]; then
    echo -e "\033[31m[BOOT] Error: Ability directory not found: $ABILITY_PACK_DIR\033[0m" >&2
    exit 1
fi

# 获取锁
LOCK_FILE="${ABILITY_PACK_DIR}/.running.lock"
exec 200>"$LOCK_FILE"
flock -x 200

# 1. 从package.yaml读取配置
PACKAGE_YAML="${ABILITY_PACK_DIR}/package.yaml"
if [ ! -f "$PACKAGE_YAML" ]; then
    echo -e "\033[31m[BOOT] Error: package.yaml not found in $ABILITY_PACK_DIR\033[0m" >&2
    exit 1
fi

# 检查yq命令是否存在
PATH="${ABILITY_PACK_DIR}/bin:$PATH"
if ! command -v yq >/dev/null 2>&1; then
    # 安装yq到bin目录
    $ABILITY_PACK_DIR/bin/start_install_yq.sh
fi

# 解析package.yaml
PYTHON_VERSION=$(yq eval '.requires-python' "$PACKAGE_YAML" | tr -d '"')
DEPENDENCIES=$(yq eval '.dependencies[]' "$PACKAGE_YAML" 2>/dev/null | tr '\n' ' ')

if [ -z "$PYTHON_VERSION" ]; then
    echo -e "\033[31m[BOOT] Error: requires-python not specified in package.yaml\033[0m" >&2
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
        echo -e "\033[31m[BOOT] Error: Failed to install Python $PYTHON_VERSION\033[0m" >&2
        exit 1
    fi
fi

# 获取Python的实际路径（解析可能的软链接）
REAL_PYTHON_PATH=$(readlink -f "$PYTHON_PATH" || echo "$PYTHON_PATH")
# echo "[BOOT] Using Python at: $REAL_PYTHON_PATH"

# 3. 配置环境
ENV_DIR="${ABILITY_PACK_DIR}/env"

if [ ! -d "$ENV_DIR" ]; then
    # 在线模式 - 使用package.yaml中的dependencies

    # echo "[BOOT] Creating virtual environment..."
    "$REAL_PYTHON_PATH" -m venv "$ENV_DIR"
    if [ -n "$DEPENDENCIES" ]; then
        # echo "[BOOT] Installing dependencies from package.yaml..."
        "${ENV_DIR}/bin/pip" install --disable-pip-version-check $DEPENDENCIES
    fi
    
    # 设置使用的Python路径为虚拟环境的Python
    PYTHON_TO_USE="${ENV_DIR}/bin/python"
    
else
    # 半离线模式 - 使用env文件夹
    ENV_PYTHON_PATH="${ENV_DIR}/bin/python"
    if [ ! -f "$ENV_PYTHON_PATH" ]; then
        echo -e "\033[31m[BOOT] Error: Python not found in env directory\033[0m" >&2
        exit 1
    fi
    
    # 仅当原环境中的python不可用时才更新软链接
    if ! "$ENV_PYTHON_PATH" --version >/dev/null 2>&1; then
        # 备份原始Python软链接（如果是软链接）
        if [ -L "$ENV_PYTHON_PATH" ]; then
            mv "$ENV_PYTHON_PATH" "${ENV_PYTHON_PATH}.bak"
        fi
        # 更新env中的Python软链接
        ln -sf "$REAL_PYTHON_PATH" "$ENV_PYTHON_PATH"
    fi
    
    # 验证Python版本
    ENV_PYTHON_VERSION=$("$ENV_PYTHON_PATH" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    REQUIRED_MIN_VERSION=$(echo "$PYTHON_VERSION" | sed 's/[>=<]//g')
    
    if [ "$(printf '%s\n' "$REQUIRED_MIN_VERSION" "$ENV_PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_MIN_VERSION" ]; then
        echo -e "\033[31m[BOOT] Error: Python version in env ($ENV_PYTHON_VERSION) does not meet requirement ($PYTHON_VERSION)\033[0m" >&2
        exit 1
    fi

    PYTHON_TO_USE="$ENV_PYTHON_PATH"
fi

# 释放锁
flock -u 200

# 激活虚拟环境
source "${ENV_DIR}/bin/activate"

# 4. 启动能力脚本

echo "[BOOT] Python=$PYTHON_TO_USE"
$PYTHON_TO_USE "${ABILITY_PACK_DIR}/src/${ABILITY_SCRIPT_ENTRY}" $1 $2
