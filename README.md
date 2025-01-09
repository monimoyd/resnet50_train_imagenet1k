# Train RestNet50 Model on Imagenet 1K
This model is a ResNet-50 trained on ImageNet dataset.

# VM Setup Steps

From AWS console create a EC2 instance of type g4dn.2xlarge.

Create a EBS volume of size 500GB and attach the volume to EC2 instance

SSH to AWS console and find the volume having 500 GB using lsblk command.
The following command shows how to mount volume /dev/nvme2n1 having 500GB 

```bash
sudo mkfs.ext4 /dev/nvme2n1
sudo mkdir /mnt/ebs
sudo mount -o rw /dev/nvme1n1 /mnt/ebs
cd /mnt/ebs
```

For activating virtual environment use the following command:

```bash
conda init --all --dry-run –verbose
conda activate pytorch
```

# Download imagenet 1K 

Install kaggle CLI

```bash
pip install kaggle
```

Login to kaggle, from settings create token and it will create kaggle.json.
Place kaggle.json in ~/.kaggle/

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

From kaggle Accept competition rules
https://www.kaggle.com/competitions/imagenet-object-localization-challenge/rules


Download ImageNet Dataset from kaggle as below:

```bash
kaggle competitions download -c imagenet-object-localization-challenge
unzip imagenet-object-localization-challenge.zip -d ILSVRC
```

Validation dataset will not be in proper directory structure, use the following command to make 
validation dataset in proper directory structure

```bash
cd /mnt/ebs/ILSVRC/Data/CLS-LOC/val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

# Training the imagenet dataset

Use the following command to train resnet50 on imagenet

```bash
python train_resnet50_cosine_annealing.py
```

# Training Methodology Used

- **Framework:** Pytorch lightning
- **Image Augmentations Used:** RandomResizedCrop, RandomHorizontalFlip, AutoAugment with IMAGENET policy
- **Optimizer:** SGD
- **Learning rate:** 0.1
- **Scheduler:** Cosine Annealing
- **Batch size:** 256
- **Maximum Number of epochs planned:** 50
- **Total training time:** ~72 hours


# Training Logs 

Achieved validation accuracy: 43.7 in 10 epochs

![Training Logs](/training_in_progress.png)


# Hugging Face Space Application link:

https://huggingface.co/spaces/Monimoy/resnet50-imagenet1k-image-classifier














