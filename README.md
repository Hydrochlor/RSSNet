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

## Datasets and Data Permissions

We provide scripts to generate the RRUFF-2Mix and UNIPR-2Mix datasets.
Before using the scripts, obtain the required input spectra and the applicable
permissions for their use. Permission to use a generation script does not
itself authorize use or redistribution of its input data or generated datasets.

To generate the corresponding dataset, run:

```generate
# generate RRUFF-2Mix dataset
bash generate_RRUFF-2Mix.sh

# generate UNIPR-2Mix dataset
bash generate_UNIPR-2Mix.sh
```

The project-provided download link for RRUFF-2Mix and UNIPR-2Mix is
[here](https://uplodea.com/AyYDIp8T8a54DjR/file). Confirm its availability and
the applicable dataset permissions before use. Download access alone does
not grant a dataset license.

RRUFF, UNIPR, RRUFF-2Mix, and UNIPR-2Mix are outside the RSSNet software
license. Their separately applicable permissions, source notices, and
upstream conditions must be followed. Where those permissions have not been
identified, neither this README nor the software license supplies them.
Project Model Weights are a separate category addressed in LICENSE; dataset
permission and model-weight permission must not be treated as interchangeable.

In the dataset, `prep_spec.txt` contains the pure spectra from the corresponding spectral library. Within each folder, the `mix_2_spec_*_*.txt` file record the indices and weight of the two pure spectra used to generate the mixed spectra, while the `*_rednoise_10-20_mix_*.txt` file contain the resulting mixed spectra.

## Training

**Before training:** If your activity creates a Research Model as defined in
[LICENSE](LICENSE), you must expressly accept the **Research Model Terms of
RSSNet License version 1.4** before starting. Read sections 2.2 and 7.3,
including the later-commercial-use and onward-transfer restrictions, mandatory
publication citation, liability limits, and law-and-court terms. A creator may
send an affirmative acceptance identifying their name/entity and the license
version to **liujinchao@nankai.edu.cn**, or use an acceptance facility provided
by the Licensor or an authorized distributor. This acceptance does not require
a separate commercial approval for otherwise permitted academic activities.
Mere access or download is not express acceptance.

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

When publicly reporting results obtained using the Software, Modifications,
or Research Models, you **must acknowledge RSSNet and cite this paper** as
required by section 7.1 of [LICENSE](LICENSE). Private study and unpublished
experiments do not require a formal paper citation.

Use your publication venue's citation format or the following BibTeX entry:

```
@misc{long2026braininspireddeepseparationnetwork,
      title={A Brain-Inspired Deep Separation Network for Single Channel Raman Spectra Unmixing}, 
      author={Gaoruishu Long and Jinchao Liu and Bo Liu and Jie Liu and Xiaolin Hu},
      year={2026},
      eprint={2604.22324},
      doi={10.48550/arXiv.2604.22324},
      archivePrefix={arXiv},
      
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.22324}, 
}
```

## License and Intellectual Property

This release uses the **RSSNet Academic Research and Non-Commercial License,
Version 1.4**. Read [LICENSE](LICENSE) for the complete terms. It is a
source-available research license with use restrictions and is not an
OSI-approved open-source license. This README summarizes the license and
does not amend it; the English license text is authoritative.

### Academic and commercial use

The covered software and Project Model Weights may be used free of charge for
Academic Non-Commercial Use as defined in LICENSE, including qualifying study,
research, teaching, and reproduction or evaluation of research results.
Project Model Weights means the original RSSNet weights supplied with this
license that the Licensor owns or is authorized to license; separately
licensed and third-party weights retain their own terms.

**Commercial Use requires a separate written agreement** from Nankai University
or its duly authorized representative before that use begins. This includes
business research, product evaluation and development, internal operations,
commercial services, and commercial model training or deployment. Research
commissioned or funded by a commercial business also requires separate written
permission unless the Licensor expressly confirms the specific research is
permitted under this license.

Ordinary academic salaries, scholarships, and government or non-profit research
grants do not by themselves make an activity commercial. University affiliation
or publication alone does not establish that the use is permitted.

### Research Models and sharing

Creating or receiving a covered Research Model requires the express acceptance
specified in section 2.2. Models trained or fine-tuned using the covered
software or weights, and models created using deliberately generated teacher
targets, pseudo-labels, or features, are subject to the definition in section 1.
Their later Commercial Use requires separate authorization even if RSSNet
code is no longer used.

Before sharing a Research Model, identify the model and its origin, provide
the Research Model Notice and the applicable terms, and obtain and retain the
recipient's express acceptance under section 7.3. An unrestricted public
download with a notice alone does not satisfy that requirement. No particular
template filename is required.

A genuinely independent retraining satisfying all exclusions in section 1 is
outside the Research Model contractual restriction. Merely resetting weights
while continuing to use RSSNet training code is not independent retraining.
Ordinary inference outputs, spectra, figures, and publications do not inherit
a non-commercial license merely because the software was used. This does not
authorize running the software or covered models for a commercial business's
needs without permission.

### Redistribution, patents, and other materials

Software and Modifications may be redistributed, including through public
research forks, subject to section 3. Include the license and required notices,
identify changes, and use the same license for Modifications. Limited recovery
of actual physical-media and delivery costs is addressed in section 3.

The project identifies Chinese patent **CN120992583B**, application
**CN202511517371.1**, *一种单通道拉曼光谱解混方法* (*A Single Channel Raman
Spectroscopy Unmixing Method*). Section 4 grants limited academic permission
under the defined Licensed Patent Claims, including the specified use of
Research Models. It does not grant rights under other patents or guarantee
current patent status, claim coverage, or non-infringement. Independent
implementations remain subject to applicable patent claims, existing
permissions, and law.

External data, third-party code and weights, papers, and other separately
licensed materials retain their applicable terms. Preserve their licenses and
notices. The software license does not override those terms or establish
missing rights; see section 6 and the data-permission explanation above.

### Citation omissions and termination

For a qualifying first inadvertent notice or citation omission, section 10.1
allows correction within 30 days after receipt of a specified written notice
to avoid termination. Even after that period, full correction automatically
restores the original permissions prospectively if there is no other uncured
breach. Intentional, repeated, commercial-use, and other breaches are subject
to section 10.2. Correction does not retroactively authorize activities during
termination. Section 10.3 explains how termination and reinstatement affect
software, patent permission, and independently owned Research Models.

### Contact

- Project contact: Jinchao Liu
- Email: liujinchao@nankai.edu.cn

The contact coordinates inquiries. An acceptance statement, an inquiry, silence,
or citation of the paper does not grant Commercial Use permission. Commercial
authorization must come from Nankai University or its duly authorized
representative in a separate written agreement.
