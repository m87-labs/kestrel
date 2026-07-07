"""Query skill leveraging the existing text generation flow."""


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..runtime import CoordToken, TextToken, Token
from kestrel.utils.spatial_refs import build_spatial_tokens, normalize_spatial_refs

from kestrel.skills.base import (
    AR_DEFAULT_MAX_NEW_TOKENS,
    AR_DEFAULT_TEMPERATURE,
    AR_DEFAULT_TOP_P,
    BuiltRequest,
    DecodeStep,
    SkillFinalizeResult,
    SkillSpec,
    SkillState,
    parse_settings,
)
from typing import Mapping

if False:  # pragma: no cover - type-checking imports
    from ..runtime import MoondreamRuntime
    from kestrel.scheduler.types import GenerationRequest


@dataclass(slots=True)
class QueryRequest:
    """Validated query payload — the carrier read by this skill's decode."""

    question: str
    image: Optional[np.ndarray | bytes]
    reasoning: bool
    stream: bool
    spatial_refs: Optional[Sequence[Sequence[float]]] = None


class QuerySkill(SkillSpec):
    """Default skill emitting plain text answers."""

    def build_request(
        self,
        image: Optional[np.ndarray | bytes],
        prompt: Mapping[str, object],
        settings: Optional[Mapping[str, object]],
    ) -> BuiltRequest:
        question = prompt.get("question")
        if question is None:
            raise ValueError("question must be provided")
        question = str(question).strip()
        if not question:
            raise ValueError("question must be a non-empty string")
        refs = normalize_spatial_refs(prompt.get("spatial_refs"))
        if refs is not None and image is None:
            raise ValueError("spatial_refs can only be used with an image")
        s = parse_settings(
            settings,
            temperature=AR_DEFAULT_TEMPERATURE,
            top_p=AR_DEFAULT_TOP_P,
            max_tokens=AR_DEFAULT_MAX_NEW_TOKENS,
        )
        request = QueryRequest(
            question=question,
            image=image,
            reasoning=bool(prompt.get("reasoning", True)),
            stream=bool(prompt.get("stream", False)),
            spatial_refs=refs,
        )
        return BuiltRequest(
            request_context=request,
            max_new_tokens=s.max_tokens,
            temperature=s.temperature,
            top_p=s.top_p,
        )

    def prompt_text(self, request_context: object) -> str:
        return getattr(request_context, "question", "")

    def __init__(self) -> None:
        super().__init__(name="query")

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        request_context: object,
    ) -> Sequence["Token"]:
        if not isinstance(request_context, QueryRequest):
            raise ValueError("QuerySkill.build_prompt_tokens requires a QueryRequest")
        prompt = request_context.question
        pt = runtime.prompt_template
        template = pt.query()
        if template is None:
            raise ValueError("Model does not include a query template")
        # Token sequence between the question and what follows it. Each model
        # declares its own opener, so per-model tokenization quirks — e.g. MD2's
        # doubled answer_id, an artifact baked into its training — stay in that
        # model's config rather than as branches in this skill.
        opener: Sequence[int] = (
            template.reasoning_prefix
            if request_context.reasoning
            else template.answer_prefix
        )
        # ``prefix_when_reasoning`` lets a model swap in a different pre-question
        # structure for CoT (Gemma 4: extra ``<|turn>system\n<|think|>`` block
        # to activate thinking). When None, the same ``prefix`` covers both.
        pre_question: Sequence[int] = (
            template.prefix_when_reasoning
            if request_context.reasoning and template.prefix_when_reasoning is not None
            else template.prefix
        )
        encoded = runtime.tokenizer.encode(prompt).ids if prompt else []
        # The runtime's _prepare_full_prefill_inputs places the first prompt token
        # before image tokens, so prepend BOS explicitly.
        tokens: List[Token] = [TextToken(token_id=int(pt.bos_id))]
        tokens.extend(TextToken(token_id=int(tid)) for tid in pre_question)
        tokens.extend(build_spatial_tokens(request_context.spatial_refs))
        tokens.extend(TextToken(token_id=int(tid)) for tid in encoded)
        tokens.extend(TextToken(token_id=int(tid)) for tid in opener)
        return tokens

    def create_state(
        self,
        runtime: "MoondreamRuntime",
        request: "GenerationRequest",
        request_context: "QueryRequest",
    ) -> "QuerySkillState":
        if not isinstance(request_context, QueryRequest):
            raise ValueError("QuerySkill.create_state requires a QueryRequest context")
        return QuerySkillState(self, request, request_context)


class QuerySkillState(SkillState):
    """Skill state that buffers tokens and exposes plain text outputs."""

    # The query mask is genuinely stateful: it transitions INACTIVE -> ACTIVE
    # mid-run. While collecting reasoning, ``allowed_token_ids`` is ``None`` and
    # (for non-moondream2 models) ``suppressed_token_ids`` is ``None`` too -- no
    # active constraint. Once the model emits ``answer_id``, the state flips to
    # forcing ``post_reasoning_prefix`` one id at a time via
    # ``allowed_token_ids``. A single spec macro-step commits a variable run
    # under ONE mask, so a run that begins in reasoning (mask = None) and crosses
    # ``answer_id`` would verify the post-boundary tokens under the stale,
    # unconstrained first-position mask -- accepting them WITHOUT the required
    # prefix constraint and corrupting the output. The scheduler's behavioural
    # fallback ("any ACTIVE constraint is stateful") cannot see this transition
    # because both masks are ``None`` at the run's first position, so declare the
    # mask stateful explicitly: the scheduler then caps such a row to one
    # committed token per macro-step, forcing exactly one constraint transition
    # per run (the regime where the per-step mask is exact). Always-cap is the
    # correctness-first verdict; a transition-only cap is not expressible here
    # because the boundary (when the model emits ``answer_id``) is model-decided
    # and unknowable before the run commits. Tradeoff (follow-up): this also caps
    # the long unconstrained answer/reasoning phases, costing intra-step
    # speculation on those rows.
    mask_is_stateful = True

    def __init__(
        self,
        spec: SkillSpec,
        request: "GenerationRequest",
        query_request: QueryRequest,
    ) -> None:
        super().__init__(spec, request)
        self._request = query_request
        self._reasoning_enabled = bool(query_request.reasoning)
        self._collecting_reasoning = self._reasoning_enabled
        self._reasoning_tokens: List[int] = []
        self._answer_tokens: List[int] = []
        self._reasoning_chunks: List[Tuple[List[int], List[Tuple[float, float]]]] = []
        self._current_chunk_tokens: List[int] = []
        self._current_chunk_points: List[Tuple[float, float]] = []
        self._pending_coord: Optional[float] = None
        self._answer_id: Optional[int] = None
        self._start_ground_id: Optional[int] = None
        self._end_ground_id: Optional[int] = None
        self._streaming = bool(query_request.stream)
        self._answer_stream_offset = 0
        # After reasoning ends (answer_id generated), replay the model's
        # post_reasoning_prefix one token at a time via allowed_token_ids.
        self._post_reasoning_tokens: Optional[List[int]] = None
        self._post_reasoning_idx: int = 0

    @property
    def emits_spatial_tokens(self) -> bool:
        # A query only consumes coord tokens while collecting *grounded*
        # reasoning (``consume_step`` folds CoordToken values into the
        # reasoning chunk's points). The answer phase and every non-reasoning
        # query are pure text -- they only handle TextToken -- so the spatial
        # head is dead weight there and the runtime skips it. Byte-identical:
        # this is exactly the phase in which a coord/size value is consumed.
        return self._collecting_reasoning

    def allowed_token_ids(
        self, runtime: "MoondreamRuntime"
    ) -> Optional[Sequence[int]]:
        if (
            self._post_reasoning_tokens is not None
            and self._post_reasoning_idx < len(self._post_reasoning_tokens)
        ):
            return [self._post_reasoning_tokens[self._post_reasoning_idx]]
        return None

    def suppressed_token_ids(
        self, runtime: "MoondreamRuntime"
    ) -> Optional[Sequence[int]]:
        if runtime.model_name != "moondream2":
            return None
        pt = runtime.prompt_template
        if self._reasoning_enabled and self._collecting_reasoning:
            # During reasoning: suppress eos and size tokens (matching HF).
            return [pt.eos_id, pt.size_id]
        # Don't suppress during post-reasoning injection (allowed_token_ids does it).
        if (
            self._post_reasoning_tokens is not None
            and self._post_reasoning_idx < len(self._post_reasoning_tokens)
        ):
            return None
        # During answer generation: suppress answer_id (matching HF).
        return [pt.answer_id]

    def consume_step(
        self,
        runtime: "MoondreamRuntime",
        step: DecodeStep,
    ) -> None:
        if self._reasoning_enabled:
            self._ensure_token_ids(runtime)
        self.append_token(step.token)

        # Track post-reasoning injection progress
        if (
            self._post_reasoning_tokens is not None
            and self._post_reasoning_idx < len(self._post_reasoning_tokens)
        ):
            self._post_reasoning_idx += 1
            # Don't collect injected tokens as answer tokens
            return None

        if not self._reasoning_enabled:
            if isinstance(step.token, TextToken):
                self._answer_tokens.append(step.token.token_id)
            return None

        if self._collecting_reasoning:
            token = step.token
            if isinstance(token, TextToken):
                token_id = token.token_id
                if token_id == self._answer_id:
                    self._collecting_reasoning = False
                    self._flush_current_chunk()
                    self._pending_coord = None
                    self._answer_stream_offset = 0
                    # Replay the model's post-reasoning opener before the answer.
                    # Empty for models without the trained-in answer_id artifact
                    # (e.g. MD3); MD2 declares [answer_id] here.
                    template = runtime.prompt_template.query()
                    self._post_reasoning_tokens = (
                        list(template.post_reasoning_prefix) if template else []
                    )
                    self._post_reasoning_idx = 0
                    return None
                if token_id == self._start_ground_id or token_id == self._end_ground_id:
                    self._flush_current_chunk()
                    self._pending_coord = None
                    return None
                self._reasoning_tokens.append(token_id)
                self._current_chunk_tokens.append(token_id)
                return None
            if isinstance(token, CoordToken):
                value = float(token.pos)
                if self._pending_coord is None:
                    self._pending_coord = value
                else:
                    self._current_chunk_points.append((self._pending_coord, value))
                    self._pending_coord = None
                return None
            # Ignore other token types during reasoning
            return None

        # Answer phase
        if isinstance(step.token, TextToken):
            self._answer_tokens.append(step.token.token_id)
        return None

    def pop_stream_delta(self, runtime: "MoondreamRuntime") -> Optional[str]:
        if not self._streaming:
            return None
        if self._collecting_reasoning:
            return None
        if not self._answer_tokens:
            return None
        text = runtime.tokenizer.decode(self._answer_tokens)
        if len(text) <= self._answer_stream_offset:
            return None
        chunk = text[self._answer_stream_offset :]
        self._answer_stream_offset = len(text)
        if not chunk:
            return None
        return chunk

    def finalize(
        self,
        runtime: "MoondreamRuntime",
        *,
        reason: str,
    ) -> SkillFinalizeResult:
        if self._reasoning_enabled:
            self._flush_current_chunk()

        tokenizer = runtime.tokenizer
        answer_text = (
            tokenizer.decode(self._answer_tokens) if self._answer_tokens else ""
        )

        output: Dict[str, Any] = {"answer": answer_text}

        if self._reasoning_enabled:
            reasoning_text = (
                tokenizer.decode(self._reasoning_tokens)
                if self._reasoning_tokens
                else ""
            )
            grounding: List[Dict[str, object]] = []
            cursor = 0
            for tokens, points in self._reasoning_chunks:
                chunk_text = tokenizer.decode(tokens) if tokens else ""
                length = len(chunk_text)
                if points:
                    grounding.append(
                        {
                            "start_idx": cursor,
                            "end_idx": cursor + length,
                            "points": points.copy(),
                        }
                    )
                cursor += length
            output["reasoning"] = {"text": reasoning_text, "grounding": grounding}

        return SkillFinalizeResult(
            text=answer_text,
            tokens=list(self.tokens),
            output=output,
        )

    def on_prefill(self, runtime: "MoondreamRuntime") -> None:
        if not self._reasoning_enabled:
            return None
        self._ensure_token_ids(runtime)
        return None

    def _flush_current_chunk(self) -> None:
        if not self._reasoning_enabled:
            return
        if self._current_chunk_tokens or self._current_chunk_points:
            self._reasoning_chunks.append(
                (
                    list(self._current_chunk_tokens),
                    list(self._current_chunk_points),
                )
            )
        self._current_chunk_tokens.clear()
        self._current_chunk_points.clear()
        self._pending_coord = None

    def _ensure_token_ids(self, runtime: "MoondreamRuntime") -> None:
        if self._answer_id is not None:
            return
        pt = runtime.prompt_template
        self._answer_id = pt.answer_id
        self._start_ground_id = pt.start_ground_points_id
        self._end_ground_id = pt.end_ground_id
