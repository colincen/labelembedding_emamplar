intent=(AddToPlaylist BookRestaurant GetWeather PlayMusic RateBook SearchCreativeWork SearchScreeningEvent)
for ((j=0; j <5; j++))
do
for ((i=0; i<${#intent[*]}; i++))
do
	python -u network_additive.py --dataset SNIPS \
	--data_dir /home/sh/data/JointSLU-DataSet/formal_snips/ \
	--bidirectional Ture \
	--lstm_dropout 0.5 \
	--num_layers 1 \
	--crf 1 \
	--dim3foratt 100 \
	--use_charEmbedding 0\
	--conv_filter_sizes \(3,4,5\) \
	--conv_filter_nums \(30,40,50\) \
	--embedding_method exemplar \
	--encoder_method wordembedding_slot_val \
	--hidden_size 300 \
	--epoch 30 \
	--log_every 20 \
	--log_valid 300 \
	--patience 5 \
	--max_num_trial 5 \
	--lr_decay 0.5 \
	--learning_rate 0.001 \
	--batch_size 32 \
	--description_path data/snips_slot_description.txt \
	--save_dir data/add_val/ --embed_file /home/sh/data/komninos_english_embeddings.gz \
	--run_type train \
	--target_domain ${intent[$i]} \
	--device cuda:1

	python -u network_additive.py --run_type test \
	--save_dir data/add_val/ \
	--target_domain ${intent[$i]} \
	--device cuda:1
done
done
