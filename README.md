# A Brain-Inspired Deep Separation Network for Single Channel Raman Spectra Unmixing

This repository is the official implementation of the following paper:

**🚀 A Brain-Inspired Deep Separation Network for Single Channel Raman Spectra Unmixing**

**✅ Accepted in International Joint Conference on Neural Networks (IJCNN 2026)**

**Authors:** Gaoruishu Long, Jinchao Liu, Bo Liu, Jie Liu & Xiaolin Hu.

## Requirements

You can use the following command to prepare your environment.

```setup
conda create -n rssnet python=3.8.10
conda activate rssnet
pip install -r requirements.txt
```

Due to the large size of the datasets, we provide code to generate the RRUFF-2Mix and UNIPR-2Mix datasets. Please run the following command to generate the corresponding dataset:

```generate
# generate RRUFF-2Mix dataset
bash generate_RRUFF-2Mix.sh

# generate UNIPR-2Mix dataset
bash generate_UNIPR-2Mix.sh
```

You can also download the datasets RRUFF-2Mix and UNIPR-2Mix from [here](https://uplodea.com/AyYDIp8T8a54DjR/file).

In the dataset, `prep_spec.txt` contains the pure spectra from the corresponding spectral library. Within each folder, the `mix_2_spec_*_*.txt` file record the indices and weight of the two pure spectra used to generate the mixed spectra, while the `*_rednoise_10-20_mix_*.txt` file contain the resulting mixed spectra.

## Training

To train the proposed RSSNet, please first modify the paths to the dataset in ```./config/train_RSSNet.yml``` so that they point to your own path.

Then, you can train your own model by using the following commands:

```train
python train_RSSNet.py --opt config/train_RSSNet.yml
```

You can also follow the same procedure to train the neural separation–based methods.

## Evaluation

To evaluate trained model on a specific dataset, run:

```eval
python test.py --opt path_to_your_model_config.yml \
               --ref path_to_prep_spec.txt \
               --list path_to_indices_and_weight.txt \
               --mix path_to_mix_spec.txt
```

If you want to change the network model used for testing, please modify the model loading section in `test.py`.

## Citation

Please cite our paper:

```
@misc{long2026braininspireddeepseparationnetwork,
      title={A Brain-Inspired Deep Separation Network for Single Channel Raman Spectra Unmixing}, 
      author={Gaoruishu Long and Jinchao Liu and Bo Liu and Jie Liu and Xiaolin Hu},
      year={2026},
      eprint={2604.22324},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.22324}, 
}
```

## License

The source code in this repository is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

Please note that datasets and third-party spectral libraries may be subject to their own licenses or terms of use.