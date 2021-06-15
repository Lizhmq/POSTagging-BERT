export CUDA_VISIBLE_DEVICES=0

python  output.py \
	--output_dir=save/1 \
        --test_name=test1.pkl \
	--input_file=data/Data1_train_utf16.tag \
	--output_file=test.txt \
	--load_dir=save/1/checkpoint-9600-0.9519
