#!/usr/bin/bash

# 确定目录参数
if [ $# -lt 1 ]; then
    read -p "Output package to: " home
else
    home=$1
fi

pdir="$home/packages/robot.test.org/0.1.0"

# 输出目录和前缀
echo "Output to ${pdir}"

# 创建必要的目录
mkdir -p ${pdir}/assets
mkdir -p ${pdir}/bin
mkdir -p ${home}/crs
mkdir -p ${pdir}/crds
mkdir -p ${pdir}/src

# 复制文件
arch=$(uname -m)
if [ "$(uname -m)" = "aarch64" ]; then
    arch="arm64"
fi

cp -r package/package.yaml ${pdir}
cp -r package/assets/* ${pdir}/assets/
cp -r package/bin/* ${pdir}/bin/
cp -r package/crds/* ${pdir}/crds/
cp -r package/crs/* ${home}/crs/
cp -r src/* ${pdir}/src/

# 如果使用离线环境则复制
if [ -d "env" ]; then
    mkdir -p ${pdir}/env
    cp -r env/* ${pdir}/env/
fi
