r"""
中文：把设置页那套「哪些区域不译」的开关，翻译成 LaTeX 解析器能用的判断。

Turn the settings page's region switches into decisions the parser can make.

The PDF path names regions with the labels a layout model emits — ``table``,
``figure_caption``, ``author`` — and a reader chooses which of them to leave in
the original. The LaTeX path answers the same question about the same regions,
and should answer it from the same switches: a reader who has said "leave the
tables alone" has said it about their papers, not about a file format.

**Where the two paths differ is in how sure they are.** The PDF path infers a
region from a bounding box and a classifier, and gets it wrong often enough to
need content rules underneath — a bibliography is recognised by reading it,
because on one paper only one reference paragraph in four carried the label.
Here nothing is inferred. An author block is whatever ``\author`` contains, a
table is what lies between ``\begin{tabular}`` and its ``\end``, and a
bibliography is written down as ``thebibliography``. The same switch is simply
more reliable on this side.

**中文：** PDF 那条路靠版面框和分类器猜区域，猜不准，所以底下还得压一层按内容判断的
规则。这边什么都不用猜：作者块就是 \author 里的东西，表格就是 \begin{tabular} 到
\end{tabular} 之间的东西，参考文献就明明白白写作 thebibliography。同一个开关，在这边
只是更可靠。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Labels this path can act on. A piece carrying anything else — a macro
# definition, a file reference — is never translated whatever the settings say,
# because there is no prose in it to translate.
# 这条路会看的标签。带别的标签的块——宏定义、文件引用——无论设置怎么写都不翻译，因为
# 里面根本没有可翻译的散文。
LABELS: tuple[str, ...] = (
    "title",
    "plain text",
    "abandon",
    "figure",
    "figure_caption",
    "table",
    "table_caption",
    "table_footnote",
    "isolate_formula",
    "formula_caption",
    "document_title",
    "author",
    "affiliation",
    "footnote",
    "reference",
    "algorithm",
)

# What is left alone when nobody has said otherwise. The same set the
# ``pdfmath`` preset carries, written out so the command line and the tests
# behave like the application without having to load its settings.
# 没人另外交代时保留原文的那些。与 pdfmath 预设相同，在这里写全，好让命令行和测试在不
# 加载应用设置的情况下也与应用一致。
DEFAULT_SKIP: frozenset[str] = frozenset({
    "abandon", "table", "table_footnote", "isolate_formula", "formula_caption",
    "author", "affiliation", "reference", "figure", "algorithm",
})


@dataclass(frozen=True)
class Selection:
    """
    中文：一份「哪些区域要译、哪些不译」的决定。

    One reader's decision about which regions to translate.

    Attributes:
        skip (frozenset[str]): Labels to leave in the original.
        translate_images (bool): Whether a figure's image file should be
            rewritten with its text translated. Follows the ``figure`` label,
            and is stated separately here because acting on it costs an OCR
            pass and rasterises a vector figure — the caller has to be able to
            check before it starts.
    """

    skip: frozenset[str] = DEFAULT_SKIP

    @property
    def translate_images(self) -> bool:
        """中文：要不要重写插图。 / Whether figure images are rewritten."""
        return "figure" not in self.skip

    def wants(self, label: str) -> bool:
        """
        中文：这个标签的内容要不要翻译。

        Whether content carrying this label should be translated.

        Args:
            label (str): One of :data:`LABELS`, or anything else.

        Returns:
            bool: ``False`` for a label the reader asked to skip, and for any
                label this path does not act on — a macro definition has no
                prose in it, so "translate everything" does not mean it.
        """
        return label in LABELS and label not in self.skip

    @classmethod
    def from_settings(cls, settings) -> Selection:
        """
        中文：按应用设置构造。

        Build from the application's settings.

        Args:
            settings (Settings): The resolved settings.

        Returns:
            Selection: Carrying the effective skip set. Selective translation
                switched off means nothing is skipped, exactly as it does for
                the PDF path.
        """
        chosen = settings.layout_filter.effective_skip_labels()
        if not settings.layout_filter.enabled:
            chosen = set()
        return cls(skip=frozenset(chosen))

    def describe(self) -> str:
        """
        中文：把这次的选择写成一行中文，好放进日志。

        Describe this selection in one line, for the log.

        Returns:
            str: The regions being left in the original, named the way the
                settings page names them.
        """
        names = {
            "title": "各级标题", "plain text": "正文", "abandon": "页眉页脚",
            "figure": "图片", "figure_caption": "图注", "table": "表格",
            "table_caption": "表注", "table_footnote": "表格脚注",
            "isolate_formula": "行间公式", "formula_caption": "公式编号",
            "document_title": "文章标题", "author": "作者",
            "affiliation": "作者单位", "footnote": "脚注", "reference": "参考文献",
            "algorithm": "伪代码 / 算法块",
        }
        kept = [names.get(label, label) for label in LABELS if label in self.skip]
        return "保留原文：" + ("、".join(kept) if kept else "无")
