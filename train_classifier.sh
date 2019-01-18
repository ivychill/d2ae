python src/classifier.py \
TRAIN \
~/datasets/lfw/lfw_mtcnnpy_160 \
~/models/facenet/20170512-110547/20170512-110547.pb \
~/models/lfw_classifier.pkl \
--batch_size 1000 \
--min_nrof_images_per_class 40 \
--nrof_train_images_per_class 35 \
--use_split_dataset