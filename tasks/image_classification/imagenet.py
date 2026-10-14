import torchvision.transforms as transforms
from datasets import load_dataset
from omegaconf import DictConfig
from tasks.image_classification.base import BaseImageClassificationTask


class ImageNet100Task(BaseImageClassificationTask):
    input_shape = (3, 224, 224)

    def setup_dataset(self, cfg: DictConfig):
        dataset = load_dataset(cfg.task.dataset)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self._build_datasets(dataset, train_transform, test_transform, splits=("train", "validation", "test"))
