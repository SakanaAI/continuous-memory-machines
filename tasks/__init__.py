from omegaconf import DictConfig

from tasks.task import Task
from tasks.image_classification import ImageNet100Task
from tasks.recall import CopyTask, AssociativeRecallTask, PrioritySortTask, ParityTask
from tasks.maze import MazeTask
from tasks.few_shot_regression import FewShotRegressionTask

__all__ = ["ImageNet100Task", "CopyTask", "AssociativeRecallTask", "PrioritySortTask", "ParityTask", "MazeTask", "FewShotRegressionTask"]

_TASKS = {
    "ImageNet100": ImageNet100Task,
    "Copy": CopyTask,
    "AssociativeRecall": AssociativeRecallTask,
    "PrioritySort": PrioritySortTask,
    "Parity": ParityTask,
    "Maze": MazeTask,
    "FewShotRegression": FewShotRegressionTask,
}


def construct_task(cfg: DictConfig, task_type: str) -> Task:
    return _TASKS[task_type](cfg)
