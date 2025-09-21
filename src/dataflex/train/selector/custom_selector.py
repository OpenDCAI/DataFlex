from dataflex.core.registry import register_selector
from dataflex.utils.logging import logger
from .base_selector import Selector

@register_selector('custom')
class CustomSelector(Selector):
    """
    一个自定义数据选择器的示例实现。
    """
    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir,
    ):
        """
        构造函数，用于初始化选择器。
        """
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        logger.info(f"CustomSelector initialized.")

    def select(self, model, step_id: int, num_samples: int, **kwargs):
        """
        核心选择逻辑。
        此方法定义了如何从数据集中选择样本。

        Args:
            model: 当前的模型。
            step_id (int): 当前的训练步数。
            num_samples (int): 需要选择的样本数量。

        Returns:
            list: 包含被选中样本索引的列表。
        """
        # 示例逻辑：简单返回从 0 到 num_samples-1 的索引列表。
        # 您可以在此实现更复杂的选择算法。
        return list(range(num_samples))