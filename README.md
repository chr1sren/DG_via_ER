### Intro

The reproduction repo for domain generalization paper: 

S. Zhao, M. Gong, T. Liu, H. Fu and D. Tao. Domain Generalization via Entropy Regularization. NeurIPS 2020. 

### Environment (for reproduction)
* Python 3.10
* torch 2.7.1+cu118
* torchvision 0.22.1+cu118

### Run

* Download the PACS dataset (https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk), and prepare the dataset as follows:
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

### Reproduction & Ablation

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
   Outputs: `figures/reproduction_plot.png`, `figures/ablation_unified.png`

4. **Paper baseline**  
   Edit `figures/paper_baseline.json` with the PACS test accuracies from the NeurIPS 2020 paper.

### Citation
```
@article{zhao2020domain,
  title={Domain Generalization via Entropy Regularization},
  author={Zhao, Shanshan and Gong, Mingming and Liu, Tongliang and Fu, Huan and Tao, Dacheng},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
