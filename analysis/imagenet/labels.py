"""Readable labels from the ImageNet-100 dataset."""


def _get_class_names(task):
    """Pull human-readable class names from the task's underlying HF dataset, or None."""
    wrapper = getattr(task, "test_dataset", None)
    if wrapper is None:
        return None
    hf = wrapper.dataset
    feat = hf.features.get("label")
    if feat is not None and getattr(feat, "names", None):
        return feat.names
    if "text" in hf.features:  # ImageNet100 stores readable labels in `text`
        labels, texts = hf["label"], hf["text"]
        mapping = {}
        for lbl, txt in zip(labels, texts, strict=True):
            if lbl not in mapping:
                mapping[lbl] = txt
        return [mapping.get(i, str(i)) for i in range(max(mapping) + 1)]
    return None
