import logging
from typing import Dict, Any, Tuple, Optional
import json

from app.core.config import settings
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

GENERIC_BLOCK_QUESTION = "Sorry could not generate response as guardrails blocked the question"
GENERIC_BLOCK_ANSWER = "Sorry could not generate response as guardrails blocked the response"

class GuardrailService:
    def __init__(self) -> None:
        self.llm_service = LLMService()
        self.default_provider = settings.default_llm_provider or "ollama"
        # Bedrock config (optional)
        self.bedrock_enabled: bool = getattr(settings, "bedrock_enabled", False)
        self.bedrock_model_id: Optional[str] = getattr(settings, "bedrock_model_id", None)
        self.bedrock_region: Optional[str] = getattr(settings, "bedrock_region", None)

    async def check_text_against_rule(
        self,
        text: str,
        rule: str,
        input_type: str = "question",
        llm_parameters: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Returns (is_allowed, decision_text). decision_text will be 'ALLOW' or 'BLOCK' when successful.
        """
        if not rule or not text:
            return True, "ALLOW"

        # Build strict classification prompt
        system_prompt = (
            "You are a strict guardrail classifier. You MUST output exactly one word: ALLOW or BLOCK. "
            "Determine if the provided input violates the given policy. If it violates, output BLOCK; "
            "otherwise output ALLOW. Do not include any explanation."
        )
        user_prompt = (
            f"Policy: {rule}\n"
            f"Type: {input_type}\n"
            f"Text: {text}\n\n"
            "Respond with only one word: ALLOW or BLOCK."
        )

        # Choose provider: default to ollama unless explicitly overridden
        effective_params = dict(llm_parameters or {})
        # provider = effective_params.get("provider", self.default_provider)

        # Force low temperature and few tokens to get deterministic ALLOW/BLOCK
        effective_params.setdefault("temperature", 0.0)
        effective_params.setdefault("max_tokens", 10)

        try:
            # For now, prefer using existing LLMService (Ollama/OpenAI/Anthropic)
            result_text = await self.llm_service.generate_response(
                query=user_prompt,
                context=[],
                system_prompt=system_prompt,
                llm_parameters=effective_params
            )

            decision = (result_text or "").strip().upper()
            if "BLOCK" in decision and "ALLOW" not in decision:
                return False, "BLOCK"
            if decision.startswith("BLOCK"):
                return False, "BLOCK"
            # Default allow unless explicitly blocked
            return True, "ALLOW"
        except Exception as e:
            logger.error(f"Guardrail check failed, defaulting to ALLOW. Error: {e}")
            return True, "ALLOW"

    async def apply_question_guardrail(
        self,
        query: str,
        guardrail_config: Dict[str, Any],
        llm_parameters: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Return (is_allowed, message). If not allowed, message is the generic block message for questions.
        """
        if not guardrail_config or not guardrail_config.get("useGuardrails"):
            return True, ""

        rule = (guardrail_config.get("questionGuardrails") or "").strip()
        if not rule:
            return True, ""

        allowed, _ = await self.check_text_against_rule(
            text=query,
            rule=rule,
            input_type="question",
            llm_parameters=llm_parameters,
        )
        if not allowed:
            return False, GENERIC_BLOCK_QUESTION
        return True, ""

    async def apply_answer_guardrail(
        self,
        answer: str,
        guardrail_config: Dict[str, Any],
        llm_parameters: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Return (is_allowed, answer_or_block_message). If not allowed, returns generic block message for answers.
        """
        if not guardrail_config or not guardrail_config.get("useGuardrails"):
            return True, answer

        rule = (guardrail_config.get("answerGuardrails") or "").strip()
        if not rule:
            return True, answer

        allowed, _ = await self.check_text_against_rule(
            text=answer,
            rule=rule,
            input_type="answer",
            llm_parameters=llm_parameters,
        )
        if not allowed:
            return False, GENERIC_BLOCK_ANSWER
        return True, answer 