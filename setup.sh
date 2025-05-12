cp -rf .ssh/ ~/
chmod 400 ~/.ssh/id_rsa

apt update
DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
apt-get install nano htop ffmpeg libsm6 libxext6 tmux git wget curl unzip -y

git config --global --add safe.directory /home/ubuntu/persistent
git config --global --add safe.directory /pvcvolume/cse290c
git config --global user.email "jonathan_wellington@icloud.com"
git config --global user.name "Joanthan Morris"

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

. ~/miniconda3/bin/activate

conda init --all