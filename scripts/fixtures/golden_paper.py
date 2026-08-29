"""Generates the synthetic fixture paper used by scripts/run_eval.py.

All facts here are invented for this fixture -- WidgetNet/WidgetBench are
not real -- see docs/superpowers/specs/2026-08-27-golden-eval-harness-design.md.
"""
from reportlab.pdfgen import canvas

PAGES = [
    "Widget Classification with Neural Networks. Abstract. We present a neural "
    "network approach for classifying industrial widgets into seven categories. "
    "Our model, WidgetNet, achieves 94.2% accuracy on the benchmark WidgetBench "
    "dataset, outperforming the previous best result by 8 percentage points. "
    "This report describes the dataset, the model architecture, and our "
    "experimental results.",

    "Dataset. The WidgetBench dataset contains 12,400 labeled widget images "
    "collected from 6 manufacturing plants between 2024 and 2025. Each image "
    "is labeled with one of seven widget categories: bolt, gear, spring, "
    "bracket, washer, valve, and hinge. The dataset is split into 9,000 "
    "training images, 1,700 validation images, and 1,700 test images.",

    "Method. WidgetNet uses a convolutional neural network with 18 layers, "
    "trained using the Adam optimizer with a learning rate of 0.0003 for 60 "
    "epochs. We apply standard data augmentation including random rotation "
    "and color jitter. Training took approximately 4 hours on a single GPU.",

    "Results. WidgetNet achieves 94.2% top-1 accuracy on the WidgetBench test "
    "set. The strongest baseline, a ResNet-34 classifier, achieves 86.1% "
    "accuracy under the same training conditions. WidgetNet's largest error "
    "category is confusing washers with brackets, accounting for 41% of all "
    "misclassifications.",

    "Limitations and Future Work. WidgetNet was evaluated only on images "
    "captured under controlled lighting conditions; performance under "
    "variable lighting is untested. Future work will explore domain "
    "adaptation techniques to improve robustness to lighting changes, and "
    "will extend the widget taxonomy beyond the current seven categories.",
]


def build_golden_paper(path: str) -> None:
    """Write the 5-page synthetic golden-eval fixture PDF to `path`."""
    c = canvas.Canvas(path)
    for page_text in PAGES:
        c.drawString(72, 720, page_text)
        c.showPage()
    c.save()
