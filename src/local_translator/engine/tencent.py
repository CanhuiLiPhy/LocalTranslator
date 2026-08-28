"""
中文：腾讯云机器翻译（TMT）后端——把段落交给 TMT，其余守卫沿用本地翻译器那一套。

Translate paragraphs through Tencent Cloud Machine Translation.

**Why this is not just another base_url.** The API mode in this project speaks
OpenAI chat completions: a system prompt, a user turn, a choice back. Tencent's
TMT is a cloud API of the older shape — one signed POST to
``tmt.tencentcloudapi.com`` carrying ``{"SourceText": ...}`` and returning
``{"Response": {"TargetText": ...}}``, authenticated with TC3-HMAC-SHA256 over a
canonical request rather than with a bearer token. Pointing ``base_url`` at it
cannot work; the protocols have nothing in common.

**What is reused.** Everything except the call itself.
:class:`~local_translator.engine.translators.TencentTranslator` subclasses the
local translator, so a paragraph still goes through glossary protection, the
formula-run merge, the placeholder check with its retry, and the fallback that
keeps the English source when a formula would otherwise be lost. Only
:meth:`build_messages` and :meth:`_request` are replaced — the seam where the
model is actually asked.

**What is different in kind.** TMT is a translation engine, not an instructed
model. It cannot be told to keep ``{v1}`` intact, to honour a glossary, or to
leave a term alone; it can only be handed text. The placeholder guard therefore
matters more here than anywhere else, and the fallback it triggers is the
honest outcome: a paragraph left in English beats a formula deleted from the
page. How often that happens is a property of the engine and is worth measuring
rather than assuming.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from datetime import UTC, datetime

import httpx

logger = logging.getLogger(__name__)

# The endpoint, service name and API version TMT is addressed by. These are
# fixed by Tencent Cloud and appear verbatim in the signature, so they are not
# settings.
# TMT 的接入点、服务名与 API 版本。这三项由腾讯云固定，且原样参与签名计算，因此不作为
# 可配置项。
_HOST = "tmt.tencentcloudapi.com"
_SERVICE = "tmt"
_VERSION = "2018-03-21"
_ACTION = "TextTranslate"
_ALGORITHM = "TC3-HMAC-SHA256"

# Languages TMT names differently from the rest of this project.
# TMT 对语言的叫法与本项目其余部分不同的那几种。
_LANGUAGES: dict[str, str] = {
    "zh-cn": "zh",
    "zh-hans": "zh",
    "zh-tw": "zh-TW",
    "zh-hant": "zh-TW",
}

# How long to wait for one translation, in seconds. TMT answers a short
# paragraph in well under a second; anything near this bound is a network fault
# rather than a slow translation.
# 单次翻译的等待上限（秒）。TMT 翻一个短段落远不到一秒，接近这个上限的一定是网络故障而
# 不是翻得慢。
_TIMEOUT = 30.0

# Retries for a rate-limited or transient failure, and the base delay between
# them. TMT's default quota is a handful of requests per second and this
# pipeline runs several paragraphs at once, so brushing the limit is expected
# rather than exceptional — it has to be waited out, not reported.
# 遇到限流或瞬时故障的重试次数与基础退避间隔。TMT 的默认配额是每秒数次，而本管线是多段
# 并发，因此擦到限流线是常态而非意外——该等，不该报错。
_RETRIES = 4
_BACKOFF = 1.5


def _language(code: str) -> str:
    """
    中文：把项目里的语言代码换成 TMT 认的写法。

    Map this project's language code onto the one TMT expects.

    Args:
        code (str): Language code as configured, e.g. ``"en"`` or ``"zh"``.

    Returns:
        str: The code TMT accepts. Unknown codes are passed through unchanged so
            a language this table has not heard of still reaches the service.
    """
    return _LANGUAGES.get((code or "").strip().lower(), (code or "").strip().lower())


def _sign(key: bytes, message: str) -> bytes:
    """
    中文：HMAC-SHA256 一步。

    One HMAC-SHA256 step of the TC3 signing chain.

    Args:
        key (bytes): Key for this step.
        message (str): Message to sign, encoded as UTF-8.

    Returns:
        bytes: The raw digest, which is the key for the next step.
    """
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def authorization(secret_id: str, secret_key: str, payload: str, stamp: int) -> str:
    """
    中文：按 TC3-HMAC-SHA256 规则算出 Authorization 头。

    Build the ``Authorization`` header Tencent Cloud's TC3 scheme requires.

    The scheme signs a canonical form of the request rather than the request
    itself, in three stages: a canonical request, a string to sign that commits
    to it, and a signing key derived from the date, region-free service name and
    a fixed terminator. Getting any byte of the canonical form wrong yields
    ``AuthFailure.SignatureFailure`` with no indication of which byte.

    Args:
        secret_id (str): Tencent Cloud SecretId. Not logged.
        secret_key (str): Tencent Cloud SecretKey. Not logged, and never leaves
            this function except as a digest.
        payload (str): The exact JSON body that will be sent, byte for byte.
        stamp (int): Unix timestamp used in both the header and the signature.

    Returns:
        str: The complete ``Authorization`` header value.
    """
    # Stage 1: the canonical request. Only POST with a JSON body is used here,
    # so the canonical URI, query string and signed headers are constant.
    # 第一步：规范请求。这里只用 POST + JSON，因此规范 URI、查询串与签名头都是常量。
    canonical = "\n".join((
        "POST",
        "/",
        "",
        f"content-type:application/json; charset=utf-8\nhost:{_HOST}\n"
        f"x-tc-action:{_ACTION.lower()}\n",
        "content-type;host;x-tc-action",
        hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    ))

    # Stage 2: the string to sign, which commits to the canonical request.
    # 第二步：待签字符串，它承诺了上面那个规范请求。
    date = datetime.fromtimestamp(stamp, tz=UTC).strftime("%Y-%m-%d")
    scope = f"{date}/{_SERVICE}/tc3_request"
    to_sign = "\n".join((
        _ALGORITHM,
        str(stamp),
        scope,
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    ))

    # Stage 3: the derived signing key, then the signature.
    # 第三步：派生签名密钥，然后算出签名。
    secret_date = _sign(f"TC3{secret_key}".encode(), date)
    secret_service = _sign(secret_date, _SERVICE)
    secret_signing = _sign(secret_service, "tc3_request")
    signature = hmac.new(
        secret_signing, to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    return (
        f"{_ALGORITHM} Credential={secret_id}/{scope}, "
        f"SignedHeaders=content-type;host;x-tc-action, Signature={signature}"
    )


def translate_text(
    text: str,
    *,
    secret_id: str,
    secret_key: str,
    region: str,
    source: str,
    target: str,
    project_id: int = 0,
    term_repos: list[str] | None = None,
    sent_repos: list[str] | None = None,
    client: httpx.Client | None = None,
) -> str:
    """
    中文：调用一次 TMT，把一段文本译成目标语言。

    Translate one piece of text through Tencent Cloud's TMT.

    Retries a rate-limited or transient failure with a growing delay. A request
    that is refused for a reason retrying cannot fix — a bad signature, an
    unsupported language pair — raises immediately, because waiting would only
    delay the same answer.

    Args:
        text (str): Source text. Sent verbatim, including any placeholders.
        secret_id (str): Tencent Cloud SecretId.
        secret_key (str): Tencent Cloud SecretKey.
        region (str): Region the request is billed against, e.g.
            ``"ap-guangzhou"``.
        source (str): Source language in TMT's spelling, or ``"auto"``.
        target (str): Target language in TMT's spelling.
        project_id (int): Tencent Cloud project id; ``0`` is the default
            project.
        term_repos (list[str] | None): Ids of custom term repositories to apply.
            TMT holds these server-side and rewrites the terms itself, so the
            text sent is untouched — unlike substituting markers into it, which
            was measured on arXiv:2608.20061 to fix eight terminology defects
            and cost 62% of the Chinese in long paragraphs. A repository is
            created in the Tencent Cloud console; there is no API for it
            (``CreateTermRepo`` answers ``InvalidAction``).
        sent_repos (list[str] | None): Ids of sentence repositories — a
            translation memory, applied the same way.
        client (httpx.Client | None): Connection pool to reuse. One is created
            per call when omitted, which is fine for a single translation and
            wasteful in a loop.

    Returns:
        str: The translated text.

    Raises:
        RuntimeError: When TMT reports an error, or when every retry was spent
            on a rate limit. The message carries TMT's own error code, which is
            what tells the reader whether to raise a quota or fix a credential.
    """
    body: dict = {
        "SourceText": text,
        "Source": source,
        "Target": target,
        "ProjectId": project_id,
    }
    if term_repos:
        body["TermRepoIDList"] = list(term_repos)
    if sent_repos:
        body["SentRepoIDList"] = list(sent_repos)
    payload = json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    owned = client is None
    session = client or httpx.Client(timeout=_TIMEOUT)
    try:
        last: str = ""
        for attempt in range(_RETRIES):
            stamp = int(time.time())
            headers = {
                "Authorization": authorization(secret_id, secret_key, payload, stamp),
                "Content-Type": "application/json; charset=utf-8",
                "Host": _HOST,
                "X-TC-Action": _ACTION,
                "X-TC-Timestamp": str(stamp),
                "X-TC-Version": _VERSION,
                "X-TC-Region": region,
            }
            response = session.post(
                f"https://{_HOST}/", content=payload.encode("utf-8"), headers=headers
            )
            body = response.json().get("Response", {})

            error = body.get("Error")
            if not error:
                return body.get("TargetText", "")

            code = error.get("Code", "")
            last = f"{code}: {error.get('Message', '')}"
            # Only a quota or a transient fault is worth waiting out. Anything
            # else will answer the same way however long we wait.
            # 只有配额与瞬时故障值得等。其余的等多久答案都一样。
            if not (code.startswith("RequestLimitExceeded")
                    or code.startswith("InternalError")
                    or code == "FailedOperation.ServiceIsolate"):
                raise RuntimeError(f"腾讯云 TMT 拒绝了这次请求（{last}）")
            delay = _BACKOFF * (2 ** attempt)
            logger.warning("TMT rate-limited (%s); waiting %.1fs", code, delay)
            time.sleep(delay)

        raise RuntimeError(f"腾讯云 TMT 连续 {_RETRIES} 次限流，已放弃（{last}）")
    finally:
        if owned:
            session.close()
