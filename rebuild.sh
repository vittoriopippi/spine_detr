git pull
docker build --build-arg USER_ID=$UID -t spine-detr . && ./launch_docker.sh