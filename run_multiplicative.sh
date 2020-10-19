intent=(AddToPlaylist BookRestaurant GetWeather PlayMusic RateBook SearchCreativeWork SearchScreeningEvent)
for((j=0; j < 10; j++))
do
for ((i=0; i<${#intent[*]}; i++))
do
	python -u network.py --dataset SNIPS \
	--data_dir /home/sh/data/JointSLU-DataSet/formal_snips/ \
	--bidirectional True \
	--lstm_dropout 0.5 \
	--num_layers 1 \
	--crf 1 \
	--use_charEmbedding 1 \
	--conv_filter_sizes \(3,4,5\) \
	--conv_filter_nums \(30,40,50\) \
	--embedding_method exemplar \
	--encoder_method wordembedding \
	--hidden_size 1206 \
	--epoch 30 \
	--attn mlp \
	--log_every 20 \
	--log_valid 300 \
	--patience 5 \
	--max_num_trial 5 \
	--lr_decay 0.5 \
	--learning_rate 0.001 \
	--batch_size 32 \
	--description_path data/snips_slot_description.txt \
	--save_dir data/multiplicative/ --embed_file /home/sh/data/glove.6B.300d.txt \
	--run_type train \
	--target_domain ${intent[$i]} \
	--device cuda:1

	python -u network.py --run_type test \
	--save_dir data/multiplicative/ \
	--attn mlp \
	--target_domain ${intent[$i]} \
	--device cuda:1
done
done
