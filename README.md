# VGG-19 Image Classification for lung disease
![]([https://github.com/rabbyfitriana/college-things/blob/main/assets/college-things-round.png](https://github.com/Abangale/Classification-VGG19/blob/main/Image/Banner.png))
This repository contains Notebook scripts for training and evaluating an image classification model `VGG19 Model.ipynb` based on the VGG-19 architecture using Tensorflow. The trained model is capable of classifying images into three categories: covid-19, normal, and pneumonia. For implementing, there is an API script that use the trained model from notebook and allows users to classify multiple images as either covid-19, normal, or pneumonia.

## Getting started

### Requirements

Before running the code, make sure the following library installed:
- Python 3.x
- PyTorch
- Tensorflow
- scikit-learn
- Matplotlib

### Training Usage

Clone the repository:

```bash
  git clone https://github.com/Abangale/Classification-VGG19.git
  ```
Prepare the Data:

- Place your datasets  inside the `LungData` directory as shown below:

  ```
  Classification-VGG19/
    Image
    Train Notebook
    classifier_api_endpoint.py
    classifier_function.py
    VGG19.pth
    LungData/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            image2.jpg
            ...
        class3/
            image1.jpg
            image2.jpg
            ...
  ```

Training and Evaluation:

- Open the `VGG19 Model.ipynb` file and modify the file directories parameters.

```python
  image_dir = Path('C:/LungData/')
  ```

Run all the Jupyter Notebook Code.

### API Usage

Open terminal then type script from below:

```bash
  uvicorn classifier_api_endpoint:app --host 0.0.0.0 --port 8000 --reload
  ```

Access the API documentation:

- Go to the web below in your web browser to access the API documentation and interact with the `/classification_predict` endpoint.

```link
http://127.0.0.1:8000/docs
```
or
```link
http://localhost:8000/docs
```

## Contributing

Contributions are welcome! If you have any suggestions or improvements for this code, please feel free to submit a pull request.

## Related projects

Here's a list of implemented related projects where you can find the original code:

- [Cat Dog Classification VGG](https://github.com/anhphan2705/Cat-Dog-Classification-VGG)
- [Lung Disease Classiffication](https://github.com/matiassingers/awesome-readme)

## Licensing

This project is licensed under the [MIT License](LICENSE).
