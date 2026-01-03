import os
import sys
import unittest

CURRENT_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from cache import LRUCache  # noqa: E402
from translator import BaseTranslator  # noqa: E402


class DummyTranslator(BaseTranslator):
    def __init__(self, cache_size: int = 2) -> None:
        super().__init__("en", cache_size=cache_size)
        self.calls = 0

    def _translate_impl(self, text, source_lang, target_lang):
        self.calls += 1
        return f"{text}:{source_lang}:{target_lang}"


class LRUCacheTests(unittest.TestCase):
    def test_lru_cache_eviction(self):
        cache = LRUCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        value, hit = cache.get("a")
        self.assertTrue(hit)
        self.assertEqual(value, 1)
        cache.set("c", 3)
        value, hit = cache.get("b")
        self.assertFalse(hit)
        self.assertIsNone(value)
        value, hit = cache.get("a")
        self.assertTrue(hit)
        self.assertEqual(value, 1)
        value, hit = cache.get("c")
        self.assertTrue(hit)
        self.assertEqual(value, 3)


class TranslatorCacheTests(unittest.TestCase):
    def test_base_translator_cache_hit(self):
        translator = DummyTranslator(cache_size=2)
        translator.translate("hello", "en")
        translator.translate("hello", "en")
        translator.translate("hello", "en")
        self.assertEqual(translator.calls, 1)

    def test_base_translator_cache_eviction(self):
        translator = DummyTranslator(cache_size=1)
        translator.translate("hello", "en")
        translator.translate("world", "en")
        translator.translate("hello", "en")
        self.assertEqual(translator.calls, 3)

    def test_base_translator_cache_disabled(self):
        translator = DummyTranslator(cache_size=0)
        translator.translate("hello", "en")
        translator.translate("hello", "en")
        self.assertEqual(translator.calls, 2)


if __name__ == "__main__":
    unittest.main()
