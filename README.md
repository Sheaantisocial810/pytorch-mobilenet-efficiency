# MobileNet: CNN Efficiency and Knowledge Distillation

This project explores CNN model efficiency through from-scratch PyTorch implementations of **MobileNetV1** and **MobileNetV2**. Its core focus is demonstrating model compression via **Knowledge Distillation (KD)**, where a large ResNet-18 (teacher) is used to train a highly efficient MobileNetV2 (student) on the CIFAR-10 dataset. This repository was developed for an M.S. Machine Learning course and includes detailed code and comparative analyses of these lightweight architectures.

## Features

* **MobileNetV1:** Full implementation from scratch using Depthwise Separable Convolutions.
* **MobileNetV2:** Full implementation from scratch, including Inverted Residuals and Linear Bottlenecks.
* **Knowledge Distillation (KD):** A complete KD training loop using a custom `DistillationLoss` (KL Divergence + Cross-Entropy) to transfer knowledge from a teacher (ResNet-18) to a student (MobileNetV2).
* **Transfer Learning:** Example of fine-tuning a model (MobileNetV1) trained on CIFAR-10 for a new task (CIFAR-100).
* **Comparative Analysis:** Scripts to directly compare MobileNetV1 vs. a standard CNN in terms of parameter count and training speed.
* **Hyperparameter Analysis:** Demonstrates the effect of the `width_multiplier` hyperparameter on the MobileNetV2 model size.

## Core Concepts & Techniques

* **Knowledge Distillation:** The core technique of model compression where a smaller "student" model is trained to mimic the output logits of a larger "teacher" model, capturing its learned nuances.
* **Model Efficiency:** Implementation of Depthwise Separable Convolutions (MobileNetV1), which dramatically reduce computational cost and parameter count compared to standard convolutions.
* **Modern CNN Architecture:** Implementation of Inverted Residuals and Linear Bottlenecks (MobileNetV2), which provide a more efficient and powerful building block for mobile-first models.
* **Transfer Learning:** The practice of "freezing" pre-trained layers and fine-tuning the final layers of a network on a new, related dataset to leverage learned features.

---

## How It Works: Concepts & Architectures

This project implements and analyzes three core concepts: the architectures of MobileNetV1 and V2, and the compression technique of Knowledge Distillation.

### 1. MobileNetV1: Depthwise Separable Convolutions

The primary innovation of MobileNetV1 is the **Depthwise Separable Convolution**. It replaces expensive standard convolutions with two more efficient operations:

1.  **Depthwise Convolution:** Applies a single spatial filter (e.g., 3x3) to *each input channel independently*. If you have $M$ input channels, this step produces $M$ output channels. It only performs spatial filtering without combining channel information.
2.  **Pointwise Convolution:** A simple 1x1 convolution that linearly combines the $M$ outputs from the depthwise step to produce $N$ new output channels. This step is responsible for feature combination across channels.

#### Mathematical Analysis (Cost Comparison)

This two-step process drastically reduces computational cost. Let's analyze the cost (number of multiplications) to produce an output feature map of size $D_F \times D_F \times N$ from an input of $D_F \times D_F \times M$ using a kernel of size $D_K \times D_K$.

* **Standard Convolution Cost:**
    The cost is the product of the output feature map size, the kernel size, and the input/output channel counts.

  $$\text{Cost}_{\text{Std}} = (D_F \cdot D_F) \cdot (D_K \cdot D_K \cdot M \cdot N)$$

* **Depthwise Separable Convolution Cost:**
    We sum the costs of the two steps.
    1.  *Depthwise Cost:* $(D_F \cdot D_F) \cdot (D_K \cdot D_K \cdot M)$
    2.  *Pointwise Cost:* $(D_F \cdot D_F) \cdot (1 \cdot 1 \cdot M \cdot N)$
    
    $$\text{Cost}_{\text{DS}} = (D_F^2 \cdot D_K^2 \cdot M) + (D_F^2 \cdot M \cdot N)$$

* **Reduction Factor:**
    The ratio of $\text{Cost}\_{\text{DS}}$ to $\text{Cost}_{\text{Std}}$ is:

  $$\frac{\text{Cost}\_{\text{DS}}}{\text{Cost}_{\text{Std}}} = \frac{D_F^2 \cdot M \cdot (D_K^2 + N)}{D_F^2 \cdot M \cdot (D_K^2 \cdot N)} = \frac{D_K^2 + N}{D_K^2 \cdot N} = \frac{1}{N} + \frac{1}{D_K^2}$$
  
    For a standard 3x3 kernel ($D_K=3$), this reduction is $\frac{1}{N} + \frac{1}{9}$. This means the computational cost is reduced by a factor of 8-9x compared to standard convolution, which is the key to MobileNet's efficiency.

### 2. MobileNetV2: Inverted Residuals & Linear Bottlenecks

MobileNetV2 improves upon V1 with two main innovations:

1.  **Inverted Residuals:** Classic residual blocks (like in ResNet) have a "wide -> narrow -> wide" structure. MobileNetV2 *inverts* this. The block first uses a 1x1 pointwise convolution to *expand* the channel dimension, then runs the 3x3 depthwise convolution on this wider representation (capturing more features), and finally uses a 1x1 pointwise convolution to *project* the features back down to a low-dimensional "bottleneck." A skip-connection connects the low-dimensional bottlenecks.
2.  **Linear Bottlenecks:** In the Inverted Residual block, the final 1x1 projection convolution *does not* have a ReLU activation. The authors found that using a non-linear activation in a low-dimensional space (the bottleneck) destroys information. By keeping this projection linear, the network's expressive power is better preserved.

The MobileNetV2 architecture in this repository is built by stacking these blocks according to the following configuration:

| $t$ (Expand Ratio) | $c$ (Output Channels) | $n$ (Repeats) | $s$ (Stride) |
|:---:|:---:|:---:|:---:|
| 1 | 16 | 1 | 1 |
| 6 | 24 | 2 | 2 |
| 6 | 32 | 3 | 2 |
| 6 | 64 | 4 | 2 |
| 6 | 96 | 3 | 1 |
| 6 | 160 | 3 | 2 |
| 6 | 320 | 1 | 1 |

### 3. Knowledge Distillation (KD)

Knowledge Distillation is a compression technique for training a small "student" model (like MobileNetV2) to mimic the performance of a large, pre-trained "teacher" model (like ResNet-18).

The student learns from two sources:
1.  **Hard Labels:** The actual ground-truth labels (e.g., `[0, 0, 1, 0, ...]`). This is trained with a standard **Cross-Entropy Loss**.
2.  **Soft Targets:** The full probability distribution from the teacher model's output. For example, the teacher might output `[0.05, 0.15, 0.7, 0.1, ...]`. This "soft" distribution contains more information than the hard label; it teaches the student *how* the teacher "thinks" (e.g., "this image is 70% a 'dog', but also 15% a 'cat'").

#### Loss Function (The "Math")

To control the "softness" of the teacher's targets, we use a **Temperature** parameter ($T$). The softmax function is modified:

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Where $z_i$ are the raw logits. When $T > 1$, this "softens" the probabilities, pushing them closer together and giving more weight to smaller logits.

The total loss function is a weighted sum of two components:

1.  **Distillation Loss ($L_{KL}$):** The **Kullback-Leibler (KL) Divergence** between the student's soft outputs ($q_s$) and the teacher's soft outputs ($q_t$), both computed with $T$. This pushes the student's probability distribution to match the teacher's.
2.  **Student Loss ($L_{CE}$):** The standard **Cross-Entropy Loss** between the student's *non-softened* outputs (logits computed with $T=1$) and the *hard* ground-truth labels ($y$).

The final combined loss is:

$$L_{KD} = \alpha \cdot (L_{KL}(q_s, q_t) \cdot T^2) + (1 - \alpha) \cdot L_{CE}(z_s, y)$$

* $\alpha$ is a hyperparameter that balances the two loss terms (e.g., $\alpha = 0.5$).
* The $T^2$ term scales the gradient of the distillation loss to ensure its magnitude is on par with the cross-entropy loss.

---

## Project Structure

```
pytorch-mobilenet-efficiency/
├── .gitignore          # Standard Python gitignore
├── LICENSE             # MIT License
├── README.md           # This readme file
├── requirements.txt    # Project dependencies
│
├── notebooks/
│   └── run_project.ipynb      # Jupyter notebook to run all scripts
│
├── scripts/
│   ├── run_01_mobilenet_v1.py               # Script to train MobileNetV1
│   ├── run_02_normal_cnn.py                 # Script to train NormalCNN
│   ├── run_03_transfer_learning.py          # Script for transfer learning
│   ├── run_04_mobilenet_v2.py               # Script to train MobileNetV2
│   ├── run_05_mobilenet_v2_hyperparams.py   # Script for WM analysis
│   └── run_06_knowledge_distillation.py     # Main script for distillation
│
└── src/
    ├── __init__.py
    ├── distillation.py        # DistillationLoss and student training loop
    │
    ├── models/
    │   ├── __init__.py
    │   ├── mobilenet_v1.py    # MobileNetV1 model definition
    │   ├── mobilenet_v2.py    # MobileNetV2 model definition
    │   └── normal_cnn.py      # NormalCNN model definition
    │
    └── utils/
    ├── __init__.py
    ├── data_loader.py    # CIFAR-10/100 data loaders
    ├── logger.py         # Logging configuration
    ├── plotting.py       # Loss plotting utility
    └── training.py       # Standard train/eval loops

````

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/pytorch-mobilenet-efficiency.git
    cd pytorch-mobilenet-efficiency
    ````

2.  **Install Dependencies:**
    It is highly recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Run an Experiment:**
    You can run any of the experiments from the `scripts/` folder. All scripts will automatically create `logs/`, `data/`, and `models/` directories to store their outputs.

    Scripts are run with default parameters. You can override any parameter using command-line arguments. To see all available options for a script, use the `-h` or `--help` flag.
    
    ```bash
    # Get help for the main distillation script
    python scripts/run_06_knowledge_distillation.py --help
    ```
    
    **Example: Run the main Knowledge Distillation experiment**
    
    ```bash
    # Run with default settings
    python scripts/run_06_knowledge_distillation.py
    
    # Run with custom settings (e.g., more epochs, higher alpha)
    python scripts/run_06_knowledge_distillation.py --epochs 20 --alpha 0.75 --lr 0.0005
    ```
    
    **Example: Run the MobileNetV1 training first** (This is required for Script 3)
    
    ```bash
    # Run with default settings
    python scripts/run_01_mobilenet_v1.py
    ```

4.  **Use the Notebook:**
    Alternatively, you can open the notebook in the `notebooks/` directory to run the scripts step-by-step from a single interface. The notebook also contains examples of how to pass custom arguments.
    
    ```bash
    jupyter notebook notebooks/run_project.ipynb
    ```

-----

## Author

Feel free to connect or reach out if you have any questions\!

  * **Maryam Rezaee**
  * **GitHub:** [@msmrexe](https://github.com/msmrexe)
  * **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

-----

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
