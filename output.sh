export CUDA_VISIBLE_DEVICES=0

python  output.py \
	--output_dir=save/1-full \
	--data_dir=./data \
        --train_name=whole_train1.pkl \
	--test_name=test1.pkl \
	--input_file=data/Data1_test_utf16.tag \
	--output_file=test1.txt \
	--load_dir=save/1-full/checkpoint-18000-0.9998
