"""Role implementations that call platform services through `aicore.client(name)`."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import Audio, BackendError, ChatResult, Msg, VoiceRef, WordTiming
from .transport import Transport

CHAT_TIMEOUT_S = 900.0
OIP_TIMEOUT_S = 600.0
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


class AicoreChatLLM:
    """`openai` / `chat.completions` on vLLM. Structured output via vLLM's `structured_outputs` (JSON schema,
    no free whitespace); thinking via chat_template_kwargs.enable_thinking (Qwen-style templates; other
    templates ignore the flag)."""

    def __init__(self, service: str, transport: Transport, *, client: Any = None):
        self.service = service
        self.transport = transport
        self._client = client

    def client(self) -> Any:
        if self._client is None:
            import aicore

            self._client = aicore.client(self.service, timeout=CHAT_TIMEOUT_S, max_retries=0)
        return self._client

    def chat(self, messages: list[Msg], *, json_schema: dict | None = None, temperature: float, seed: int,
             max_tokens: int, thinking: bool | None = None) -> ChatResult:
        body: dict[str, Any] = {"messages": [m.to_dict() for m in messages], "temperature": temperature,
                                "seed": seed, "max_tokens": max_tokens}
        extra: dict[str, Any] = {}
        if json_schema is not None:
            # vLLM structured outputs; without disable_any_whitespace a model can pad the JSON with whitespace
            # until max_tokens (seen with Gemma inside arrays)
            extra["structured_outputs"] = {"json": json_schema, "disable_any_whitespace": True}
        if thinking is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": thinking}

        def call() -> Any:
            import aicore

            resp = self.client().chat.completions.with_raw_response.create(
                model=aicore.model_name(self.service), extra_body=extra or None, **body)
            return resp.http_response.json()

        raw = self.transport.call(self.service, {**body, **extra}, call)
        return parse_chat(raw, self.service)


def parse_chat(raw: dict[str, Any], service: str) -> ChatResult:
    try:
        choice = raw["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError, TypeError):
        raise BackendError(f"{service}: malformed chat response: {str(raw)[:300]}") from None
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    m = re.search(r"<think>(.*?)</think>", content, re.S)
    if m:
        reasoning = (reasoning or "") + m.group(1)
        content = _THINK.sub("", content)
    elif "</think>" in content:  # template opened <think> in the prompt
        head, _, content = content.partition("</think>")
        reasoning = (reasoning or "") + head
    return ChatResult(text=content.strip(), reasoning=(reasoning or None), finish_reason=choice.get("finish_reason"),
                      usage=raw.get("usage") or {}, model=service)


class _OIP:
    def __init__(self, service: str, transport: Transport, *, client: Any = None):
        self.service = service
        self.transport = transport
        self._client = client

    def client(self) -> Any:
        if self._client is None:
            import aicore

            self._client = aicore.client(self.service, timeout=OIP_TIMEOUT_S)
        return self._client

    def infer(self, inputs: dict[str, np.ndarray], request: dict[str, Any],
              parameters: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        """Text (BYTES) inputs must also appear in `request`: only numeric arrays go into the recording key."""

        def call() -> Any:
            out = self.client().infer(inputs, parameters=parameters) if parameters else self.client().infer(inputs)
            return {k: {"dtype": str(v.dtype), "shape": list(v.shape), "data": v.reshape(-1).tolist()}
                    for k, v in out.items()}

        numeric = {k: v for k, v in inputs.items() if v.dtype != object}
        req = {**request, "parameters": parameters} if parameters else request
        raw = self.transport.call(self.service, req, call, arrays=numeric)
        return {k: np.asarray(v["data"], dtype=v["dtype"]).reshape(v["shape"]) for k, v in raw.items()}


class AicoreEnhancer(_OIP):
    """deepfilternet3 / wpe-dereverb: AUDIO FP32 [-1], SAMPLE_RATE INT32 [1] -> AUDIO, SAMPLE_RATE."""

    def clean(self, audio: Audio) -> Audio:
        out = self.infer({"AUDIO": audio.samples.astype(np.float32), "SAMPLE_RATE": np.array([audio.sr], np.int32)},
                         {"op": "clean"})
        return Audio(out["AUDIO"].astype(np.float32), int(out["SAMPLE_RATE"][0]))


class AicoreQualityScorer(_OIP):
    """utmos: AUDIO FP32 [-1, -1] (zero-padded), LENGTHS INT32 [-1], SAMPLE_RATE INT32 [1] -> MOS FP32 [-1].
    One clip per request for now (§4.2 notes); the signature already takes batches."""

    def score(self, clips: Sequence[Audio]) -> np.ndarray:
        scores = []
        for clip in clips:
            out = self.infer({"AUDIO": clip.samples[None, :].astype(np.float32),
                              "LENGTHS": np.array([len(clip.samples)], np.int32),
                              "SAMPLE_RATE": np.array([clip.sr], np.int32)}, {"op": "score"})
            scores.append(float(out["MOS"].reshape(-1)[0]))
        return np.asarray(scores, dtype=np.float32)


class AicoreVoiceCloner(_OIP):
    """voxcpm2: TEXT BYTES [1], REF_AUDIO FP32 [-1], REF_SAMPLE_RATE INT32 [1], REF_TEXT BYTES [1], SEED INT64 [1]
    -> AUDIO FP32 [-1], SAMPLE_RATE INT32 [1]. `params` go out as request parameters (mode, cfg_value,
    inference_timesteps); stage 4 sets them from `voice`."""

    def __init__(self, service: str, transport: Transport, *, params: dict[str, Any] | None = None, client: Any = None):
        super().__init__(service, transport, client=client)
        self.params = dict(params or {})

    def clone(self, text: str, ref: VoiceRef, *, seed: int) -> Audio:
        inputs = {"TEXT": np.array([text], dtype=object), "REF_AUDIO": ref.audio.samples.astype(np.float32),
                  "REF_SAMPLE_RATE": np.array([ref.audio.sr], np.int32), "REF_TEXT": np.array([ref.text], dtype=object),
                  "SEED": np.array([seed], np.int64)}
        out = self.infer(inputs, {"op": "clone", "text": text, "ref_text": ref.text}, self.params or None)
        return Audio(out["AUDIO"].reshape(-1).astype(np.float32), int(out["SAMPLE_RATE"].reshape(-1)[0]))


ASR_SR = 16000
_QWEN_ASR_PREFIX = re.compile(r"^\s*language\s+[^<]*<asr_text>", re.I)


def parse_transcription(raw: Any) -> str:
    """`{"text": ...}` (json) or a bare string; Qwen3-ASR's "language X<asr_text>..." prefix is removed."""
    text = raw.get("text", "") if isinstance(raw, dict) else str(raw or "")
    return _QWEN_ASR_PREFIX.sub("", text).strip()


class AicoreTranscriber:
    """`openai` / `audio.transcriptions` (vLLM Qwen3-ASR, or a custom server such as meralion-3-3b-asr). The clip is
    sent as a 16 kHz PCM16 WAV; both ASR models work at 16 kHz."""

    def __init__(self, service: str, transport: Transport, *, client: Any = None):
        self.service = service
        self.transport = transport
        self._client = client

    def client(self) -> Any:
        if self._client is None:
            import aicore

            self._client = aicore.client(self.service, timeout=OIP_TIMEOUT_S, max_retries=0)
        return self._client

    def transcribe(self, audio: Audio, *, language: str | None = None) -> str:
        a = audio.resample(ASR_SR)
        wav = a.wav_bytes()

        def call() -> Any:
            import aicore

            kw = {"language": language} if language else {}
            resp = self.client().audio.transcriptions.with_raw_response.create(
                model=aicore.model_name(self.service), file=("audio.wav", wav, "audio/wav"), response_format="json", **kw)
            return resp.http_response.json()

        raw = self.transport.call(self.service, {"op": "transcribe", "language": language}, call,
                                  arrays={"AUDIO": a.samples})
        return parse_transcription(raw)


class AicoreSpeakerEmbedder(_OIP):
    """wavlm-sv / redimnet2-sv: AUDIO FP32 [-1, -1], LENGTHS INT32 [-1], SAMPLE_RATE INT32 [1] -> EMBEDDINGS FP32
    [-1, -1] (L2-normalised by the service; normalised again here). One clip per request for now."""

    def __init__(self, service: str, transport: Transport, *, name: str, client: Any = None):
        super().__init__(service, transport, client=client)
        self.name = name

    def embed(self, clips: Sequence[Audio]) -> np.ndarray:
        rows = []
        for clip in clips:
            out = self.infer({"AUDIO": clip.samples[None, :].astype(np.float32),
                              "LENGTHS": np.array([len(clip.samples)], np.int32),
                              "SAMPLE_RATE": np.array([clip.sr], np.int32)}, {"op": "embed"})
            v = out["EMBEDDINGS"].reshape(-1).astype(np.float32)
            rows.append(v / (np.linalg.norm(v) or 1.0))
        return np.stack(rows) if rows else np.zeros((0, 0), np.float32)


class AicoreAligner(_OIP):
    """qwen3-forced-aligner: AUDIO FP32 [-1], SAMPLE_RATE INT32 [1], TEXT BYTES [1], LANGUAGE BYTES [1]
    -> WORDS BYTES [-1], TIMES FP32 [-1, 2], SCORES FP32 [-1]."""

    def align(self, audio: Audio, text: str, *, language: str) -> list[WordTiming]:
        out = self.infer({"AUDIO": audio.samples.astype(np.float32), "SAMPLE_RATE": np.array([audio.sr], np.int32),
                          "TEXT": np.array([text], dtype=object), "LANGUAGE": np.array([language], dtype=object)},
                         {"op": "align", "text": text, "language": language})
        words = [str(w) for w in out["WORDS"].reshape(-1)]
        times = out["TIMES"].reshape(-1, 2)
        scores = out["SCORES"].reshape(-1)
        return [WordTiming(w, float(t[0]), float(t[1]), float(s)) for w, t, s in zip(words, times, scores, strict=True)]
