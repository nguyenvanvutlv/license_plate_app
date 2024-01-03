NOTE:
python3.9.13
- Tải tất cả các thư viện có trong file requirements.txt
- Yêu cầu phải có GPU để chạy được model
- chạy python main.py để chạy chương trình

# GUIDE

## SYSTEM

<table style="margin: left">
  <tr>
    <th align="left">FIELD</th>
    <th align="left">CONTENT</th>
  </tr>
  <tr>
    <td>OS Name</td>
    <td>Ubuntu 22.04.3 LTS</td>
  </tr>
  <tr>
    <td>Processor</td>
    <td>AMD® Ryzen 7 5800hs with radeon graphics × 16</td>
  </tr>
  <tr>
    <td>Graphics</td>
    <td>NVIDIA Corporation GA106M [GeForce RTX 3060 Mobile / Max-Q]</td>
  </tr>

</table>


## INSTALLATION


> if using Pip

```bash
pip install -r requirements.txt (PIP)
```

> if using anaconda  from scratch

```bash
conda create -n <name> python=3.9
conda install -c anaconda numpy -y
conda install -c anaconda opencv -y
conda install -c anaconda tk -y
conda install -c anaconda pillow -y

# if machine have only support CPU
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# if machine support GPU
# cuda version in this project 12.x
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

```

## RUN


- Go to release v2.0.0 and download model and checkpoint file, follow the folder structure bellow

```
model  
│
└───detection
│   └───best.pt
│   
└───recognition
    └───ocr.pth
``````

```bash
python main.py
```

# VERSION (v2.0.0)

[Release new version](https://github.com/nguyenvanvutlv/license_plate_app/releases/tag/v2.0.0)