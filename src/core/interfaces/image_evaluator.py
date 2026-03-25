import abc
import tensorflow as tf


class ImageEvaluator(abc.ABC):

    @abc.abstractmethod
    def select_model(self) -> dict:
        pass

    @abc.abstractmethod
    def get_model_info(self) -> dict:
        pass

    @abc.abstractmethod
    def evaluate_image(
        self, raw_img: tf.Tensor | list[tf.Tensor]
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
        pass

    pass


class AsyncImageEvaluator(abc.ABC):

    @abc.abstractmethod
    async def select_model(self) -> dict:
        pass

    @abc.abstractmethod
    async def get_model_info(self) -> dict:
        pass

    @abc.abstractmethod
    async def evaluate_image(
        self, raw_img: tf.Tensor | list[tf.Tensor]
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
        pass

    pass
