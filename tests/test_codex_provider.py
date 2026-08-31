import unittest
import unittest.mock
from src.generation.deepseek_client import (
    _map_codex_reasoning_effort,
    get_provider_api_key,
    get_provider_base_url,
    chat,
    CODEX_AVAILABLE,
)
from src.utils.token_tracker import log_response

if CODEX_AVAILABLE:
    from openai_codex.api import ReasoningEffort
else:
    ReasoningEffort = None


class TestCodexProvider(unittest.TestCase):
    def test_reasoning_effort_mapping(self):
        if not CODEX_AVAILABLE:
            self.skipTest("openai_codex not installed")

        self.assertEqual(_map_codex_reasoning_effort(True), ReasoningEffort.high)
        self.assertEqual(_map_codex_reasoning_effort(False), ReasoningEffort.none)
        self.assertEqual(_map_codex_reasoning_effort("none"), ReasoningEffort.none)
        self.assertEqual(_map_codex_reasoning_effort("low"), ReasoningEffort.low)
        self.assertEqual(_map_codex_reasoning_effort("medium"), ReasoningEffort.medium)
        self.assertEqual(_map_codex_reasoning_effort("high"), ReasoningEffort.high)
        self.assertEqual(_map_codex_reasoning_effort("xhigh"), ReasoningEffort.xhigh)
        self.assertEqual(_map_codex_reasoning_effort("max"), ReasoningEffort.xhigh)

    @unittest.mock.patch("src.generation.deepseek_client.Codex")
    def test_chat_codex_execution(self, mock_codex_cls):
        if not CODEX_AVAILABLE:
            self.skipTest("openai_codex not installed")

        mock_session = unittest.mock.MagicMock()
        mock_codex_cls.return_value.__enter__.return_value = mock_session

        mock_thread = unittest.mock.MagicMock()
        mock_session.thread_start.return_value = mock_thread

        mock_result = unittest.mock.MagicMock()
        mock_result.error = None
        mock_result.final_response = "Generated Response from Codex"
        mock_result.usage = unittest.mock.MagicMock()
        mock_result.usage.input_tokens = 120
        mock_result.usage.output_tokens = 350
        mock_result.usage.reasoning_tokens = 50
        mock_thread.run.return_value = mock_result

        response = chat(
            prompt="Tạo một câu hỏi thi",
            system="Bạn là chuyên gia khảo thí",
            model="gpt-5.6-luna",
            provider="codex",
            thinking="low",
        )

        self.assertEqual(response, "Generated Response from Codex")
        mock_session.thread_start.assert_called_once()
        start_kwargs = mock_session.thread_start.call_args[1]
        self.assertEqual(start_kwargs.get("model"), "gpt-5.6-luna")
        self.assertEqual(start_kwargs.get("developer_instructions"), "Bạn là chuyên gia khảo thí")

        mock_thread.run.assert_called_once()
        run_kwargs = mock_thread.run.call_args[1]
        self.assertEqual(run_kwargs.get("effort"), ReasoningEffort.low)

    def test_token_tracker_turn_result(self):
        mock_turn = unittest.mock.MagicMock()
        mock_turn.usage = unittest.mock.MagicMock()
        mock_turn.usage.input_tokens = 100
        mock_turn.usage.output_tokens = 200
        mock_turn.usage.reasoning_tokens = 40

        # Should not raise exception
        log_response(mock_turn, model="gpt-5.6-luna")


if __name__ == "__main__":
    unittest.main()
