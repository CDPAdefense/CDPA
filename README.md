### Leakage-Resilient and Carbon-Neutral Aggregation Featuring the Federated AI-enabled Critical Infrastructure

Here, we implement CDPA against ROG attacks.
<img src="defense.jpg" width=600>
#### Prerequisites

- install Python packages
    ```bash
    pip3 install -r requirements.txt
    ```
- Download the pretrained models and put them under `model_zoos` ([link](https://huggingface.co/erickyue/rog_modelzoo/tree/main)) （These models retrived from ROG attack）

- The images for a minimal runnable example has been included under `data` folder. The ImageNet validation dataset can be used for a full test.   

- Run the example with FedAvg:
    ```bash
    python attack_fedavg.py
    ```
    The script will load the configuration file `config_fedavg.yaml`. 


    You can change the settings in the configuration file. 