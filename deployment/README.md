# Corner Detection Deployment

## Description

In order to deploy the corner detection model efficiently, we have prepared an api using FastAPI. In order to infer an
image, an api is developed which get an image as input and displays an output image after inference procedure.

---

### Usage

To deploy CornerDetection model, follow these steps:

1. Clone the repository to your local machine
2. Prepare path of a directory which consists of config file and checkpoints
3. Run the code with below command:

```
python app.py --result-path=[path_to_result_directory] --checkpoint_name=[name_of_checkpoint(ex:best_model.pth)]
```

4. Upload an image


