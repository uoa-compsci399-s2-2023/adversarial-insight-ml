<p align="center">
  <a href="https://uoa-compsci399-s2-2023.github.io/adversarial-insight-ml/">
    <img
      src="./docs/_static/img_logo.png"
      alt="AIML Logo"
      style="
        width: 418px;
        height: 418px;
        max-width: 100%;
        height: auto;
      "
    >
  </a>
</p>

<h2 align="center">Adversarial Insight ML (AIML)</h2>

<p align="center">
  <a href="https://pypi.org/project/adversarial-insight-ml/">
    <img src="https://badge.fury.io/py/adversarial-insight-ml.svg" alt="PyPI version" />
  </a>
 <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.9-blue.svg" alt="Python version" />
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style" />
  </a>
  <a href="https://uoa-compsci399-s2-2023.github.io/adversarial-insight-ml/">
    <img src="https://img.shields.io/badge/Documentation-Click%20Here-blue.svg" alt="Documentation" />
  </a>
</p>


> “Why does your machine lie?”

AIML, short for Adversarial Insight ML, is your go-to Python package for gauging your machine learning image classification models' resilience against adversarial attacks. With AIML, you can effortlessly test your models against a spectrum of adversarial examples, receiving crisp, insightful feedback. AIML strives for user-friendliness, making it accessible even to non-technical users.

We've meticulously selected various attack methods, including AutoProjectedGradientDescent, CarliniL0Method, CarliniL2Method, CarliniLInfMethod, DeepFool, PixelAttack, SquareAttack, and ZooAttack, to provide a strong robustness assessment.

For more information, visit our [PyPI page](https://pypi.org/project/adversarial-insight-ml/) and [documentation page](https://uoa-compsci399-s2-2023.github.io/adversarial-insight-ml/).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install Adversarial Insight ML, you can use pip:

```bash
pip install adversarial-insight-ml
```

## Usage

Here's a simple overview of AIML in use:
<p align="center">
  <img src="./docs/_static/img_workflow.png" alt="img_workflow.png" width="390" height="360" align="center">
</p>

You can evaluate your model with the `evaluate` function:

```python
from aiml.evaluation.evaluate import evaluate

evaluate(model, test_dataset)
```

The `evaluate` function has **two required parameters**:
- `input_model (str or model)`: A string of the name of the machine learning model or the machine learning model itself.
- `input_test_data (str or dataset)`: A string of the name of the testing dataset or the testing dataset itself.

The `evaluate` function has the following **optional parameters**:
- `input_train_data (str or dataset, optional)`: A string of the name of the training dataset or the training dataset itself (default is None).
- `input_shape (tuple, optional)`: Shape of input data (default is None).
- `clip_values (tuple, optional)`: Range of input data values (default is None).
- `nb_classes (int, optional)`: Number of classes in the dataset (default is None).
- `batch_size_attack (int, optional)`: Batch size for attack testing (default is 64).
- `num_threads_attack (int, optional)`: Number of threads for attack testing (default is 0).
- `batch_size_train (int, optional)`: Batch size for training data (default is 64).
- `batch_size_test (int, optional)`: Batch size for test data (default is 64).
- `num_workers (int, optional)`: Number of workers to use for data loading (default is half of the available CPU cores).
- `dry (bool, optional)`: When True, the code should only test one example.
- `attack_para_list (list, optional)`: List of parameter combinations for the attack.

See the demos in `examples/` directory for usage in action:
- [demo_basic.ipynb](examples/demo_basic.ipynb)
- [demo_huggingface.ipynb](examples/demo_huggingface.ipynb)
- [demo_robustbench.ipynb](examples/demo_robustbench.ipynb)

## Features

After evaluating your model with `evaluate` function, we provide
the following insights:

- Summary of adversarial attacks performed, found in a text file named `attack_evaluation_result.txt` followed by date. For example:
  ![img_result.png](./docs/_static/img_result.png)
- Samples of the original and adversarial images can be found in a directory named `img/` followed by date. For example:
  <p align="center">
    <img src="./docs/_static/img_folderlist.png" alt="img_folderlist.png" width="420"> 
    <img src="./docs/_static/img_sample.png" alt="img_sample.png" width="500" align="center">
  </p>
    

## Contributing

**Code Style**  
Always adhere to the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for writing Python code. Allow upto 99 characters per line as the absolute maximum. Alternatively, just use [black](https://github.com/psf/black).

**Commit Messages**  
As per [Documentation/SubmittingPatches](https://git.kernel.org/pub/scm/git/git.git/tree/Documentation/SubmittingPatches?h=v2.36.1#n181) in the Git repo, write commit messages in present tense and imperative mood, e.g., "Add feature" instead of "Added feature". Craft your messages as if you're giving orders to the codebase to change its behaviour.

**Branching**  
We conform to a variation of the "GitHub Flow'' convention, but not strictly. For example, see the following types of branches:

- main: This branch is always deployable and reflects the production state.
- bugfix/\*: For bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We extend our sincere appreciation to the following individuals who have been instrumental in the success of this project:

Firstly, our client [Luke Chang](https://github.com/changx03). His invaluable guidance and insights guided us from the beginning through every phase, ensuring our work remained aligned with practical needs. This project would not have been possible without his efforts.

We'd also like to express our gratitude to [Dr. Asma Shakil](https://github.com/asma-shakil), who has coordinated and provided an opportunity for us to work together on this project.

Thank you for being part of this journey.

Warm regards,
Team 7

## Team 7 - Contacts

- Sungjae Jang (Team Leader)
  - GitHub: [eaj3](https://github.com/eaj3)
  - Email: sjan260@aucklanduni.ac.nz
- Takuya Saegusa
  - GitHub: [TakuyaSaegusa](https://github.com/TakuyaSaegusa)
  - Email: tsae032@aucklanduni.ac.nz
- Haozhe Wei
  - GitHub: [Cyber77UoA](https://github.com/Cyber77UoA)
  - Email: hwei313@aucklanduni.ac.nz
- Yuming Zhou
  - GitHub: [zhou920ming](https://github.com/zhou920ming)
  - Email: yzho739@aucklanduni.ac.nz
- Terence Zhang
  - GitHub: [Tyranties](https://github.com/Tyranties)
  - Email: tzha820@aucklanduni.ac.nz
