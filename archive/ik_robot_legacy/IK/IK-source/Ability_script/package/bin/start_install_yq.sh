#!/bin/bash

# 获取系统架构
ARCH=$(uname -m)
case $ARCH in
    x86_64)
        ARCH="amd64"
        ;;
    aarch64|arm64)
        ARCH="arm64"
        ;;
    armv7l|armv6l)
        ARCH="arm"
        ;;
    *)
        echo -e "\033[31m[INSTALL YQ] Unsupported architecture: $ARCH\033[0m"
        exit 1
        ;;
esac

# 下载URL
DOWNLOAD_URL="https://github.com/mikefarah/yq/releases/latest/download/yq_linux_${ARCH}"
echo "[INSTALL YQ] Downloading yq from $DOWNLOAD_URL"

# 安装目录
INSTALL_DIR="${ABILITY_PACK_DIR}/bin"
BINARY_NAME="yq"

# 临时目录
TMP_DIR=$(mktemp -d)

if ! wget -q "$DOWNLOAD_URL" -O "$TMP_DIR/$BINARY_NAME"; then
    echo -e "\033[31m[INSTALL YQ] Failed to download yq from $DOWNLOAD_URL\033[0m"
    rm -rf "$TMP_DIR"
    exit 1
fi

# 设置可执行权限
chmod +x "$TMP_DIR/$BINARY_NAME"

# 移动二进制文件
mv "$TMP_DIR/$BINARY_NAME" "$INSTALL_DIR/$BINARY_NAME"

# 清理临时文件
rm -rf "$TMP_DIR"

# 验证安装
if ! command -v "$INSTALL_DIR/$BINARY_NAME" >/dev/null 2>&1; then
    echo -e "\033[31m[INSTALL YQ] yq installation failed.\033[0m"
    exit 1
else
    echo "[INSTALL YQ] $($INSTALL_DIR/$BINARY_NAME --version)"
    exit 0
fi