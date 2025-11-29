<!-- insertion marker -->
<a name="0.1.0"></a>

## [0.1.0](https://github.com///compare/2b4b12998170fd44bbc92919c24a6a61f2249163...0.1.0) (2025-11-30)

### Features

- implement data preparation function with timing and preprocessing ([bb114a6](https://github.com///commit/bb114a6e06656396afe0f6664783990e15cef3ae))
- add data preprocessing functionality in preprocessor.py ([e7e2d07](https://github.com///commit/e7e2d07549bd5313efbbbda689617d3da336cef9))
- add build_word2id_seqs utility to module exports ([e0d6dae](https://github.com///commit/e0d6daee0657a4cdf3b4a7757ae6d5a07503bd65))
- enhance tokenization functions and add build_word2id_seqs utility ([91bc446](https://github.com///commit/91bc446e425492a255199db1348f6b3f20981020))
- add new file paths for data sources in cfg_base.py ([9b19772](https://github.com///commit/9b197724545f487fd341a1d13182acb0ff588b67))
- add setter.py for data preparation functionality ([4c57ce0](https://github.com///commit/4c57ce0b3c943ea1b9328b8c47c5065954b59603))
- add processor.py for data processing functionality ([81cbac8](https://github.com///commit/81cbac8eff2d1e7e80b170b950d491112a032d7d))
- add argument parsing and data preparation import in main.py ([3d3666e](https://github.com///commit/3d3666e0a9a0909ae2a140247d5c916e0ab26d03))
- add __init__.py for data processing module in deep learning workflow ([12685e1](https://github.com///commit/12685e1cf8ba0b2ead865b26bac098fc9934a13c))
- add CHANGELOG.md for project versioning and feature tracking ([0f2aaf5](https://github.com///commit/0f2aaf51890f36789d8f13058416a24d6bac0c46))
- add weibo_comments.csv as nlp project data ([71d64fb](https://github.com///commit/71d64fbd497d60356e808008d9531e3a36455c33))
- add uv.lock ([647198d](https://github.com///commit/647198d110712f11b6a36701bf570ccd79905c04))
- add unet_sem_seg.py for binary segmentation evaluation and training ([ec3b63b](https://github.com///commit/ec3b63bf5ea44162fe2243b8cc6da03e612d61c7))
- add unet_sem_seg.py for custom PyTorch Dataset class for semantic segmentation ([89627b4](https://github.com///commit/89627b482e797fb0672a10efd7bae43164666df3))
- add unet_mask_mapper.py for mask to index conversion in binary segmentation ([5d5b6cd](https://github.com///commit/5d5b6cd934fe97dbc361e208cc4769b87853091a))
- add unet_focal.py for Focal Loss implementation in binary segmentation ([e63eeab](https://github.com///commit/e63eeab4ee7c497e0b85532c38617a02514f332f))
- add unet_edge.py for Edge-Aware Loss implementation ([3a7c548](https://github.com///commit/3a7c548373a05f3f041469f84410632b62dddb72))
- add unet_dnf.py for Dice-Focal-BCE loss implementation ([07d40e2](https://github.com///commit/07d40e289eb7a50db07a1dd132e398dbc07ce10e))
- add unet_dice.py for Dice loss implementation combining BCE and Dice coefficients ([ea4fe71](https://github.com///commit/ea4fe716411bd92f3f5cb7eb224cbeaacb7386ac))
- add unet_5layers_sem_seg.py for 5-layer UNet model implementation ([f886be0](https://github.com///commit/f886be03f476c4a94694191bdb0b4c39737f09d1))
- add unet_4layers_sem_seg.py for 4-layer UNet model implementation ([0c548fa](https://github.com///commit/0c548fafc42c3d9f232cb2e8cb33963d65fb312a))
- add THU.py for text segmentation and POS tagging using THULAC ([1e183c5](https://github.com///commit/1e183c5c08511eb9f6432cca4700891fccc7c79c))
- add stats.py for data processing and analysis utilities ([757f6d0](https://github.com///commit/757f6d03dbae6c6fb910cda78b29987792950293))
- add TorchDataset4SeqPredictionNextStep for next-step sequence prediction ([496022b](https://github.com///commit/496022b268c52d835c1919580371e89b190d2ac1))
- add TorchTrainer4Seq2Classification for training and validating sequential models ([83bb39e](https://github.com///commit/83bb39ede503cd0e26680eee8fab37d192f89912))
- add TorchDataset4Seq2Classification for handling sequential data in PyTorch ([d276554](https://github.com///commit/d276554d077013f930ccaaa69e6249771b20188f))
- add LSTMRNNForClassification model for multi-class classification tasks ([5aead0a](https://github.com///commit/5aead0a30b4491eadaec3ed79e9b415663e3e6a1))
- add reshaper.py for reshaping flattened tensors to grayscale ([d37b330](https://github.com///commit/d37b330e9d2492920bc1bcaece8b228bed46762b))
- add requirements.txt for project dependencies ([13921d0](https://github.com///commit/13921d00be2d5205e198aefc42689a79f6a7502b))
- add Chinese README for project overview, features, and environment setup instructions ([3373920](https://github.com///commit/337392055191c17a1ceecd7dbf8809b02150ed3b))
- add comprehensive README with project overview, features, and environment setup instructions ([b6d671b](https://github.com///commit/b6d671bd46ba2b09f1887ccf8a74ac2358568745))
- update pyproject.toml with project description and dependencies ([fbfcfa2](https://github.com///commit/fbfcfa26fd34867ef87230e041a02b5a47e0c578))
- add PT.py for managing random seed and device checks in PyTorch ([25f8c1d](https://github.com///commit/25f8c1db70567d734d1b7f0d65ff2183f96722cb))
- add predictor.py with main function template ([1a999f0](https://github.com///commit/1a999f0245e23f025b81530682eba2b0de4ae183))
- add nlp.py for Chinese and English text processing with SpaCy ([007f446](https://github.com///commit/007f44652cb578a189ac8a41417d154ea14e8e6a))
- add TorchTrainer class for managing regression model training and validation ([2547c91](https://github.com///commit/2547c913bef51646a3e2eeb060f5493b29316e1f))
- add regression Log Mean Squared Error loss function for regression tasks ([8ba472a](https://github.com///commit/8ba472a9f18f2abfc16e3e5680f6d57253e51042))
- add custom PyTorch Dataset class for label classification ([beb0641](https://github.com///commit/beb064171003a6c082e4137d414a4d7b06c8d846))
- add command-line argument parsing for UNet training configuration ([3f1791a](https://github.com///commit/3f1791ae6084e23afc5cd7ff95f076142cc87798))
- add logging functionality with configurable log file and console output ([9f60ce2](https://github.com///commit/9f60ce214536182f38f06c33ba509347886854a5))
- add text highlighting functions and formatting utilities ([a396276](https://github.com///commit/a396276e0729cc3d145c52a8c7f8c7bfa4136122))
- add helper functions and context managers for beautification, timing, and random seed management ([0358fe1](https://github.com///commit/0358fe17d8cc9b5a0de39c080848af6fb04f06cf))
- add decorators for function output beautification and timing ([dce3e0d](https://github.com///commit/dce3e0d1fbf25c7268c4d52126101a80a70643bf))
- add configuration classes for UNet parameters and preprocessing ([0780975](https://github.com///commit/07809759f0d5c41fda168546f8cd9f7a3ef72cbc))
- add configuration classes for RNN parameters and preprocessing ([42d63f8](https://github.com///commit/42d63f8ecefeedfcd8a0f583e52d3ac13aee18f9))
- add configuration classes for MLP parameters and preprocessing ([22e9695](https://github.com///commit/22e96950cb63d74029abfe70cecce32390937871))
- add configuration classes for CNN parameters and preprocessing ([9e9d789](https://github.com///commit/9e9d789556f21f687f2eb4640ef7607c68238286))
- add configuration classes for data preprocessing and hyperparameters ([ed4a7f8](https://github.com///commit/ed4a7f8f6863a7be75dd0bcfc7fccfaba356bf6f))
- add configuration module with file paths and settings ([bf2919b](https://github.com///commit/bf2919bfcafd99d2991d4cbf4438782dd3aedca2))
- add custom TorchDataLoader class for efficient data handling ([74540d2](https://github.com///commit/74540d23b225f725e761f12a43f8e37405099bc4))
- add API wrappers for OpenAI and DeepSeek text and image completion ([b9564a1](https://github.com///commit/b9564a1c13f4f2df66075f3bbea5fb7309a4143f))
- add Trainers module for specialized PyTorch training frameworks ([f4b49db](https://github.com///commit/f4b49db0e74be92452fb1ad7928c19ca157c0163))
- add utility module for comprehensive ML and data processing functions ([ac4d2fc](https://github.com///commit/ac4d2fc49b4b46de092276550727f5a222d950db))
- add Trainers module with specialized PyTorch training implementations ([570f2bf](https://github.com///commit/570f2bfeb9894f2af49aa4d07548342e53ef25dc))
- add neural network module with architectures for segmentation and classification ([cbd7c2f](https://github.com///commit/cbd7c2f862b731d58865d92fcdf8171aa8f12a1f))
- add Trainer module for neural network training implementations ([4ae0024](https://github.com///commit/4ae00241f324ddfd9c45990d63ec0301d2f628e0))
- add Dataloader module with specialized PyTorch DataLoader wrappers ([9f0ab10](https://github.com///commit/9f0ab10915c175b18a5e0f78908a51035b5e7ea6))
- add criterion module with specialized PyTorch loss functions ([ee38b87](https://github.com///commit/ee38b87eca606c4b2b3350611ad399975bdb74bf))
- add initial configuration module for ML/Data Processing ([c29fb30](https://github.com///commit/c29fb30ac6c26ee5ccbfd3104f7481ab40613343))

### Bug Fixes

- comment out ALPHA hyperparameter in cfg_base4dl.py ([15799bd](https://github.com///commit/15799bd95ddbfcfc30dd413ef05b456121dfa984))
- update import from processor to preprocessor in __init__.py ([66c4c67](https://github.com///commit/66c4c672939c46e455779bcf156d954e9b2b1610))

### Chore

- add .gitignore to exclude Python-generated files and IDE configurations ([995f430](https://github.com///commit/995f430b6980ef5552a1ec14d79276a5629c58b6))
- update CHANGELOG.md with recent feature additions ([571e9e9](https://github.com///commit/571e9e97afd12dab1c3bcd60c1586c6a7dfe6f08))

### Code Refactoring

- remove TODO comment for further data preparation steps in setter.py ([2759ecb](https://github.com///commit/2759ecbc97b07654bc98a2157cb9efd9b8c95ac3))
- change name of processor.py ([052f2ac](https://github.com///commit/052f2ac26f95767626a4a11bc44bb907d8955c49))

