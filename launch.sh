cd /home/user/workspace
chown user logs/
chown user spine_detr/
su - user

export CUDA_VISIBLE_DEVICES=$1
python3 main.py --dataset_file 2d_spine --coco_path fake_coco --num_queries 30 --spine_folder "/home/data/Diagnostikbilanz/Fertig 20190503/" --batch_size 32 --comment "$2" $3