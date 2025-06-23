from unittest.mock import MagicMock
from src.app import translate

def test_translate_basic_text():
    # Mock tokenizer and model
    tokenizer = MagicMock()
    model = MagicMock()

    tokenizer.encode.return_value = "encoded_input"
    model.generate.return_value = ["generated_output"]
    tokenizer.decode.return_value = "Shakespearean translation"

    # Fake torch.no_grad context manager
    class DummyContext:
        def __enter__(self): return None
        def __exit__(self, *args): return False

    import torch
    torch.no_grad = DummyContext

    result = translate("Bonjour", tokenizer, model)
    assert result == "Shakespearean translation"
