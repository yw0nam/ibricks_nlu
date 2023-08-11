CUDA_VISIBLE_DEVICES=0,1 python train_huggingface.py --model_address skt/kogpt2-base-v2 --model_save_path ./models_zoo/skt_kogpt2-base-v2
CUDA_VISIBLE_DEVICES=0,1 python train_huggingface.py --model_address kykim/gpt3-kor-small_based_on_gpt2 --model_save_path ./models_zoo/kykim_gpt3-kor-small_based_on_gpt2
CUDA_VISIBLE_DEVICES=1 python train_huggingface.py --model_address EleutherAI/polyglot-ko-1.3b --model_save_path ./models_zoo/EleutherAI_polyglot-ko-1.3b --batch_size_per_device 4
CUDA_VISIBLE_DEVICES=1 python train_huggingface.py --model_address skt/ko-gpt-trinity-1.2B-v0.5 --model_save_path ./models_zoo/skt_ko-gpt-trinity-1.2B-v0.5 --batch_size_per_device 16
