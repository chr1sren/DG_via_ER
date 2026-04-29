### Intro

The reproduction repo for domain generalization paper: 

S. Zhao, M. Gong, T. Liu, H. Fu and D. Tao. Domain Generalization via Entropy Regularization. NeurIPS 2020. 

### Environment (for reproduction)
* Python 3.10
* torch 2.7.1+cu118
* torchvision 0.22.1+cu118

### Dataset prepare

* Download the PACS dataset (https://www.kaggle.com/datasets/nickfratto/pacs-dataset), and prepare the dataset as follows:
    ```
    |- dataset
        |- PACS
            |- art_painting
                |- *.jpg
            |- cartoon
                |- *.jpg
            |- sketch
                |- *.jpg
            |- photo
                |- *.jpg
    ```

### Run

1. **Reproduction**  
   ```bash
   python run_all.py reproduction
   ```

2. **Ablation**  
   ```bash
   python run_all.py ablation_all
   ```
   To run with less epoch, e.g: 50 epoch:
   ```bash
   python run_all.py ablation_all --epoch 50
   ```

3. **Plot**  
   ```bash
   python run_all.py plot
   ```

4. **Paper baseline**  
   Edit `figures/paper_baseline.json` with the PACS test accuracies from the NeurIPS 2020 paper.
