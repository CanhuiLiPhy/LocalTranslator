# 第三方组件与许可证

本项目按 **AGPL-3.0** 发布：运行时导入并扩展 BabelDOC、直接调用 PyMuPDF，两者均为
AGPL-3.0，本项目构成衍生作品，须采用同一许可证。若将本项目部署为他人可通过网络访问的
服务，须依 AGPL 向使用者提供包含修改在内的完整源码；本机自用不受此约束。

## 决定许可证的组件

| 组件 | 版本 | 许可证 | 用途 |
| --- | --- | --- | --- |
| [BabelDOC](https://github.com/funstory-ai/BabelDOC) | 0.6.4 | AGPL-3.0 | PDF 解析、公式占位、重排与输出管线 |
| [PyMuPDF](https://github.com/pymupdf/PyMuPDF) | 1.28.2 | AGPL-3.0（或 Artifex 商业授权） | 插图与矢量图还原、超链接还原、目录页重绘 |

本项目不修改、不再分发 BabelDOC 的源码：它作为依赖安装，若干类与方法在运行时被替换。
补丁全部在 `src/local_translator/engine/`，各模块文档字符串注明所替换的符号。

## 借鉴而非依赖

以下项目的代码未被复制、也未作为依赖引入，本项目按其思路或行为重新实现：

| 项目 | 许可证 | 借鉴内容 | 实现位置 |
| --- | --- | --- | --- |
| [LaTeXTrans](https://github.com/NiuTrans/LaTeXTrans) | MIT | LaTeX 源码的拆分与还原顺序 | `src/local_translator/latex/` |
| [OpenCV](https://github.com/opencv/opencv) | Apache-2.0 | `resize` 等九个函数的行为约定 | `engine/cvshim.py`（纯 numpy） |
| [scikit-image](https://github.com/scikit-image/scikit-image) / [scikit-learn](https://github.com/scikit-learn/scikit-learn) | BSD-3-Clause | SSIM 默认参数；`min_samples=1` 的 DBSCAN | `engine/scishim.py` |
| [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate) | AGPL-3.0 | 选择性翻译预设（不翻脚注、姓名、表格） | 预设配置 |

OpenCV 与 scikit 系从源码安装时仍会随 BabelDOC 装上，但翻译进程在导入 BabelDOC 前
即以 shim 替换；安装包（DMG）不含这四个包。

## 随仓库分发的数据与字体

### 术语库 `src/local_translator/glossary/glossary.tsv`

8,255 条，`source` 列逐条标明来源：

| 来源 | 条数 | 许可证 |
| --- | --- | --- |
| [Wikidata](https://www.wikidata.org) 实体标签 | 5,738 | CC0 1.0 |
| [机器之心 · 人工智能术语库](https://github.com/jiqizhixin/Artificial-Intelligence-Terminology-Database) | 2,392 | CC BY-NC-SA 4.0 |
| 本项目整理（builtin / manual） | 125 | AGPL-3.0 |

源自机器之心的条目经筛选与校订，仍按 **CC BY-NC-SA 4.0** 提供：须署名，衍生数据须以
同一许可共享，**不得商用**。本项目免费分发，不构成商业使用；商业场景须先删除
`source` 列为 `jiqizhixin` 的行，其余条目不受影响。

### 词频表 `src/local_translator/glossary/common-english.txt`

派生自 [google-10000-english](https://github.com/first20hours/google-10000-english)
（前 9,999 词，个别英式拼写改为美式），源头为 LDC 发布的 Google Web Trillion Word
Corpus。上游允许教育与个人/研究用途，商业用途建议向 LDC 取得授权。本项目仅用作术语
清洗时的常用词排名。

### 汇文仿宋 `src/local_translator/assets/fonts/HuiwenFangSong.ttf`

作者知乎 [@特里王](https://zhuanlan.zhihu.com/p/12669052378)，译文正文默认中文字体。授权声明全文：

> **授权声明**
>
> - 本系列字体（包括汇文明朝体、汇文仿宋、汇文正楷、汇文港黑四款）可免费用于印刷、显示、出版、影视，或嵌入系统、软件、文档、设备。
> - 本系列字体，除作者本人在手机商城售卖外，不会授权给任何第三方渠道收费提供／分发，且严禁以任何方式将本系列字体作为单独商品或资源包的一部分出售盈利。
> - 本系列字体可以随意转发、复制。
> - 本系列字体允许进行修改或制作衍生版本。
>
> 特此声明。

本项目将该字体嵌入免费分发的软件，属第一条允许的用途，不单独提供或出售该字体。

## 其余 Python 依赖

均为宽松许可：

| 组件 | 版本 | 许可证 |
| --- | --- | --- |
| NiceGUI | 3.16.0 | MIT |
| FastAPI / Starlette | 0.141.1 / 1.6.0 | MIT / BSD-3-Clause |
| Uvicorn | 0.52.3 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| openai | 3.1.0 | Apache-2.0 |
| pydantic / pydantic-settings | 2.13.4 / 2.15.0 | MIT |
| typer / click | 0.27.1 / 8.4.2 | MIT / BSD-3-Clause |
| rich | 15.0.0 | MIT |
| keyring | 25.7.0 | MIT |
| tqdm | 4.70.0 | MPL-2.0 AND MIT |
| onnxruntime | 1.28.0 | MIT |
| huggingface-hub | 1.27.0 | Apache-2.0 |
| numpy | 2.5.2 | BSD-3-Clause 等 |
| Pillow | 12.3.0 | MIT-CMU |
| PyYAML | 6.0.3 | MIT |
| tenacity | 9.1.4 | Apache-2.0 |
| orjson | 3.12.0 | MPL-2.0 AND (Apache-2.0 OR MIT) |
| pywebview（可选） | 6.2.1 | BSD-3-Clause |

安装包由 PyInstaller 6.22.2 构建（GPL-2.0，附允许打包产物按任意许可分发的例外条款）。

## 运行时下载的资源

不随仓库分发，安装或首次使用时按需下载：

| 资源 | 何时下载 | 许可证 |
| --- | --- | --- |
| llama.cpp 预编译二进制 | `localtrans setup`（mini/full） | MIT |
| Hunyuan Hy-MT2 1.8B / 7B GGUF 权重 | `localtrans setup`（mini/full） | 腾讯混元社区许可（见下） |
| DocLayout-YOLO 版面模型 | BabelDOC 首次运行 | AGPL-3.0 |
| 思源宋体 Source Han Serif | BabelDOC 首次运行 | SIL OFL 1.1 |
| PP-OCR 识别模型 | 图片翻译首次使用 | Apache-2.0 |
| pdf.js | `localtrans setup-reader` | Apache-2.0 |
| Tectonic（或随 `-tex` 安装包内置） | 构建时 | MIT |
| ECDICT 词典数据 | 词典功能首次设置 | MIT |

Hy-MT2 权重按腾讯混元社区许可（Tencent HY Community License）提供：授权地域不含
欧盟、英国与韩国；须遵守其可接受使用政策；产品月活超一亿须另行申请授权。该许可仅
约束权重，不影响本项目代码；位于上述地区的用户请改用云端 API 后端。

## 商标

BabelDOC、PyMuPDF、Tencent Hunyuan、机器之心及其它出现于此的名称属于各自权利人，
本项目与它们没有隶属或背书关系。
