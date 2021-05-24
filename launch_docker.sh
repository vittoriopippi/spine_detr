export UID=$(id -u)
export GID=$(id -g)
docker run -v "/home/data/Diagnostikbilanz/Fertig 20190503":"/home/data/Diagnostikbilanz/Fertig 20190503" -v /home/vpippi/projects/ostfalia_detr/spine_plot:/workspace/spine_plot -v /home/vpippi/projects/ostfalia_detr/logs:/workspace/logs --user $UID:$GID --workdir="/home/$USER" --gpus all -it spine-detr