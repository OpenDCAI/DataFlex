import warnings
import unittest

from dataflex.core.registry import Registry


class RegistryBuildTests(unittest.TestCase):
    def test_build_rejects_unknown_cfg_keys(self):
        registry = Registry()

        @registry.register("selector", "strict")
        class StrictSelector:
            def __init__(self, dataset, cache_dir, seed=42):
                self.dataset = dataset
                self.cache_dir = cache_dir
                self.seed = seed

        with self.assertRaisesRegex(ValueError, "Unknown config parameter"):
            registry.build(
                "selector",
                "strict",
                runtime={"dataset": "train-set", "unused_runtime": "ignored"},
                cfg={"cache_dir": "/tmp/cache", "seedd": 7},
            )

    def test_build_keeps_runtime_injection_but_filters_unknown_runtime(self):
        registry = Registry()

        @registry.register("mixer", "runtime")
        class RuntimeMixer:
            def __init__(self, dataset, cache_dir, accelerator=None):
                self.dataset = dataset
                self.cache_dir = cache_dir
                self.accelerator = accelerator

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = registry.build(
                "mixer",
                "runtime",
                runtime={"dataset": "train-set", "accelerator": "acc", "unused_runtime": "ignored"},
                cfg={"cache_dir": "/tmp/cache"},
            )

        self.assertEqual(obj.dataset, "train-set")
        self.assertEqual(obj.cache_dir, "/tmp/cache")
        self.assertEqual(obj.accelerator, "acc")
        self.assertFalse(hasattr(obj, "unused_runtime"))

    def test_build_warns_on_unknown_runtime_keys(self):
        registry = Registry()

        @registry.register("mixer", "runtime-warning")
        class RuntimeMixer:
            def __init__(self, dataset, cache_dir, accelerator=None):
                self.dataset = dataset
                self.cache_dir = cache_dir
                self.accelerator = accelerator

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj = registry.build(
                "mixer",
                "runtime-warning",
                runtime={"dataset": "train-set", "accelerator": "acc", "unused_runtime": "ignored"},
                cfg={"cache_dir": "/tmp/cache"},
            )

        self.assertEqual(obj.dataset, "train-set")
        self.assertEqual(len(caught), 1)
        self.assertIn("Ignoring unknown runtime parameter(s)", str(caught[0].message))
        self.assertIn("'unused_runtime'", str(caught[0].message))

    def test_skip_strict_validation_opt_out(self):
        registry = Registry()

        @registry.register("weighter", "loose")
        class LooseWeighter:
            skip_strict_validation = True

            def __init__(self, strategy="uniform", **kwargs):
                self.strategy = strategy
                self.extra = kwargs

        obj = registry.build(
            "weighter",
            "loose",
            runtime={},
            cfg={"strategy": "custom", "typo_but_allowed": True},
        )

        self.assertEqual(obj.strategy, "custom")

    def test_build_reports_component_with_no_configurable_parameters(self):
        registry = Registry()

        @registry.register("selector", "no-config")
        class NoConfigSelector:
            def __init__(self):
                pass

        with self.assertRaisesRegex(ValueError, "has no configurable parameters"):
            registry.build("selector", "no-config", runtime={}, cfg={"typo": True})


if __name__ == "__main__":
    unittest.main()
