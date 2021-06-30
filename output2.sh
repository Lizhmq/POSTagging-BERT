export CUDA_VISIBLE_DEVICES=2

python  output.py \
	--output_dir=save/2-full \
	--data_dir=./data \
        --train_name=whole_train2.pkl \
	--test_name=test2.pkl \
	--input_file=data/Date2_test_utf16.tag \
	--output_file=test2.txt \
	--load_dir=save/2-full/checkpoint-22000-0.9998
