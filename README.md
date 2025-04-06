<p align="center">
    <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" align="center" width="30%">
</p>
<p align="center"><h1 align="center">NEURAL_COLLABORATIVE_FILTERING</h1></p>

<p align="center">
	<img src="https://img.shields.io/github/license/LI-SUJU/neural_collaborative_filtering?style=plastic&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/LI-SUJU/neural_collaborative_filtering?style=plastic&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/LI-SUJU/neural_collaborative_filtering?style=plastic&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/LI-SUJU/neural_collaborative_filtering?style=plastic&color=0080ff" alt="repo-language-count">
</p>
<p align="center">Built with the tools and technologies:</p>
<p align="center">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=plastic&logo=Python&logoColor=white" alt="Python">
</p>
<br>

## 🔗 Table of Contents

- [📍 Overview](#-overview)
- [👾 Dataset](#-dataset)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
  - [☑️ Prerequisites](#-prerequisites)
  - [⚙️ Installation](#-installation)
  - [🤖 Usage](#🤖-usage)
- [🔰 Contributing](#-contributing)
- [🎗 License](#-license)

---

## 📍 Overview

This project implements a Neural Collaborative Filtering (NCF) model using PyTorch, based on the MovieLens 1M dataset. 
The model combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) branches to capture both linear and 
non-linear user-item interactions for personalized recommendation.

---

## 👾 Dataset

- **MovieLens 1M**  
  Download link: [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)
  
Place the dataset file `ratings.dat` into the `dataset/` directory.

---

## 📁 Project Structure

```sh
└── neural_collaborative_filtering/
    ├── NCF.py
    ├── NCF_degraded.py
    ├── README.md
    ├── dataset
    │   ├── .DS_Store
    │   ├── README
    │   ├── movies.dat
    │   ├── ratings.dat
    │   └── users.dat
    ├── loss_curve_32_32_32_32.png
    ├── loss_curve_[128_128].png
    ├── loss_curve_[128_63_32]_negatives_2.png
    ├── loss_curve_[128_63_32]_negatives_3.png
    ├── loss_curve_[32_32].png
    ├── loss_curve_[32_32_32_32].png
    └── loss_curve_model_32_32.png
```


---
## 🚀 Getting Started

### ☑️ Prerequisites

Before getting started with neural_collaborative_filtering, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python


### ⚙️ Installation

Install neural_collaborative_filtering using one of the following methods:

**Build from source:**

1. Clone the neural_collaborative_filtering repository:
```sh
❯ git clone https://github.com/LI-SUJU/neural_collaborative_filtering
```

2. Navigate to the project directory:
```sh
❯ cd neural_collaborative_filtering
```

3. Install the project dependencies:

```bash
pip install torch pandas numpy matplotlib
```



### 🤖 Usage
Run neural_collaborative_filtering using the following command:
```bash
python NCF.py
```
### Outputs
- Training and validation loss curves saved as `.png`
- Test metrics printed in the console

---

## 🔰 Contributing

- **💬 [Join the Discussions](https://github.com/LI-SUJU/neural_collaborative_filtering/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/LI-SUJU/neural_collaborative_filtering/issues)**: Submit bugs found or log feature requests for the `neural_collaborative_filtering` project.
- **💡 [Submit Pull Requests](https://github.com/LI-SUJU/neural_collaborative_filtering/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/LI-SUJU/neural_collaborative_filtering
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/LI-SUJU/neural_collaborative_filtering/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=LI-SUJU/neural_collaborative_filtering">
   </a>
</p>
</details>

---

## 🎗 License

This project is protected under the MIT License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

