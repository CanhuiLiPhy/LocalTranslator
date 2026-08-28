#!/bin/bash
# 中文：一键安装——检查环境、装依赖、选版本、跑配置向导、生成 macOS 应用。
# One-command install: check the environment, install dependencies, pick an
# edition, run the configuration wizard and build the macOS app bundle.
#
# Everything this does is reversible and stays inside the user's home directory:
# dependencies go into a project-local .venv, configuration into
# ~/Library/Application Support/LocalTranslator, the app bundle into
# ~/Applications. Nothing is written outside $HOME and nothing needs sudo —
# the optional hostname step is a separate script precisely because it does.
# 本脚本的每一步都可撤销，且只写入用户主目录：依赖装进项目内的 .venv，配置写到
# ~/Library/Application Support/LocalTranslator，应用包放到 ~/Applications。
# 全程不写 $HOME 以外的位置、不需要 sudo——需要 sudo 的域名步骤刻意单独成一个脚本。
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
RED=$'\033[31m'; RESET=$'\033[0m'

step()  { printf '\n%s▸ %s%s\n' "$BOLD" "$1" "$RESET"; }
ok()    { printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$1"; }
warn()  { printf '  %s!%s %s\n' "$YELLOW" "$RESET" "$1"; }
die()   { printf '  %s✗%s %s\n' "$RED" "$RESET" "$1" >&2; exit 1; }
ask()   { local r; read -r -p "  $1 [$2] " r; echo "${r:-$2}"; }

printf '%sLocal Translator 安装程序%s\n' "$BOLD" "$RESET"
printf '%s%s%s\n' "$DIM" "$PROJECT" "$RESET"

# --------------------------------------------------------------------------- #
# 1. 环境检查 / Environment
# --------------------------------------------------------------------------- #
step "检查环境"

[ "$(uname -s)" = "Darwin" ] || die "本脚本针对 macOS；其它系统请按 README 手动安装。"
ok "macOS $(sw_vers -productVersion) ($(uname -m))"

if ! xcode-select -p >/dev/null 2>&1; then
  warn "未安装 Xcode 命令行工具，部分依赖需要它来编译。"
  echo "  运行：xcode-select --install，装完后重新执行本脚本。"
  exit 1
fi
ok "Xcode 命令行工具"

# uv manages both the Python version and the virtual environment, so the system
# Python is never touched — which is why it is the one thing installed here.
# uv 同时管理 Python 版本与虚拟环境，因此完全不动系统 Python——这也是这里唯一需要
# 安装的东西。
if command -v uv >/dev/null 2>&1; then
  ok "uv $(uv --version | awk '{print $2}')"
else
  warn "未找到 uv（本项目用它管理 Python 与依赖）。"
  echo "  安装命令：curl -LsSf https://astral.sh/uv/install.sh | sh"
  if [ "$(ask '现在安装 uv？(y/n)' 'y')" != "y" ]; then
    die "请先安装 uv 后重试。"
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv 安装后仍未找到，请重开终端再试。"
  ok "uv 安装完成"
fi

# --------------------------------------------------------------------------- #
# 2. 选择版本 / Edition
# --------------------------------------------------------------------------- #
step "选择版本"
cat <<'TXT'
    1. 完整版     — 本地 Hy-MT2 7B + 云端 API + 混合模式
                    约 7 GB 磁盘（llama.cpp 运行时 + 7B 权重 + 1.8B 草稿模型），
                    建议 Apple Silicon 且 16 GB 以上内存。可完全离线。
    2. 本地轻量版 — 本地 Hy-MT2 1.8B + 云端 API + 混合模式
                    约 1.5 GB 磁盘。8 GB 内存的 Mac 也能跑，同样可完全离线；
                    正文质量接近 7B，但长段落与公式密集段更依赖云端。
    3. 轻量版     — 只用云端 API
                    不下载任何模型权重，任何 Mac 都能跑，需要一个
                    OpenAI 兼容的 API key。
TXT
EDITION_CHOICE="$(ask '选择 (1/2/3)' '3')"
case "$EDITION_CHOICE" in
  1) EDITION=full ;;
  2) EDITION=mini ;;
  3) EDITION=lite ;;
  *) die "无效选择：$EDITION_CHOICE" ;;
esac
printf '%s\n' "$EDITION" > src/local_translator/edition.txt
ok "已选择：$EDITION"

# --------------------------------------------------------------------------- #
# 3. 依赖 / Dependencies
# --------------------------------------------------------------------------- #
step "安装依赖"
NATIVE="$(ask '安装独立桌面窗口支持？不装则用浏览器打开 (y/n)' 'y')"
if [ "$NATIVE" = "y" ]; then
  uv sync --extra native
else
  uv sync
fi
ok "依赖就绪（虚拟环境在 $PROJECT/.venv）"

# --------------------------------------------------------------------------- #
# 4. 配置 / Configuration
# --------------------------------------------------------------------------- #
step "配置"
LOCALTRANS_EDITION="$EDITION" uv run localtrans init

# --------------------------------------------------------------------------- #
# 5. 自检 / Verify
# --------------------------------------------------------------------------- #
step "自检"
LOCALTRANS_EDITION="$EDITION" uv run localtrans doctor || true

step "完成"
cat <<TXT
  启动方式（任选其一）：
    · uv run localtrans serve                    浏览器界面
    · uv run localtrans app                      桌面窗口
    · uv run localtrans translate 某篇论文.pdf     直接翻译一篇
    · uv run localtrans latex 2302.13971         从 LaTeX 源码翻译

  想要可双击的应用，请到 Releases 页面下载安装包。

  重新配置：uv run localtrans init
  切换版本：编辑 src/local_translator/edition.txt（full / mini / lite），
            改完如果换到有本地模型的版本，再跑一次 uv run localtrans setup
TXT
