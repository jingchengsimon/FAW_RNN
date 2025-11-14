#!/bin/bash

# 使用说明：
# 1. 将下面的 YOUR_GITHUB_USERNAME 替换为你的 GitHub 用户名
# 2. 将 REPOSITORY_NAME 替换为你的仓库名称（例如：aim3_RNN）
# 3. 运行此脚本：bash push_to_github.sh

# 配置信息（请修改以下两行）
GITHUB_USERNAME="jingchengsimon"
REPOSITORY_NAME="FAW_RNN"

# 或者直接使用完整的仓库 URL（取消注释下面一行并填入你的仓库 URL）
# REPO_URL="https://github.com/YOUR_GITHUB_USERNAME/aim3_RNN.git"

# 如果使用 REPO_URL，取消注释下面这行
# REMOTE_URL="$REPO_URL"

# 如果使用用户名和仓库名，使用下面这行
REMOTE_URL="https://github.com/${GITHUB_USERNAME}/${REPOSITORY_NAME}.git"

echo "正在添加远程仓库: $REMOTE_URL"
git remote add origin "$REMOTE_URL" 2>/dev/null || git remote set-url origin "$REMOTE_URL"

echo "正在推送代码到 GitHub..."
git branch -M main 2>/dev/null || git branch -M master
git push -u origin $(git branch --show-current)

echo "完成！代码已推送到 GitHub。"

