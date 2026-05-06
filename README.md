# Flower Hub Benchmark

This directory presents five realistic federated learning (FL) benchmark tasks: [medical image segmentation](./fed-med-seg/), [financial fraud detection](./fed-fin-fraud/), [legal instruction tuning](./fed-legal-llm/), [phishing URL detection](./fed-phish-guard/), and [on-device audio tagging](./fed-audio-tagging/).

These tasks involve sensitive, distributed data across institutions or user devices, making FL a natural framework for collaborative training. Together, the benchmarks cover both cross-silo and cross-device settings and span multiple data modalities, including images, tabular data, audio, and text. For each task, we provide a complete training and evaluation pipeline and use it to benchmark widely adopted aggregation algorithms under standardized experimental settings.

For more details, see the individual application directories. All benchmark applications are also available on [Flower Hub](https://flower.ai/apps/): [fed-med-seg](https://flower.ai/apps/flwrlabs/fed-med-seg/), [fed-fin-fraud](https://flower.ai/apps/flwrlabs/fed-fin-fraud/), [fed-legal-llm](https://flower.ai/apps/flwrlabs/fed-legal-llm/), [fed-phish-guard](https://flower.ai/apps/flwrlabs/fed-phish-guard/), [fed-audio-tagging](https://flower.ai/apps/flwrlabs/fed-audio-tagging/).
