# demo
python3 scripts/01_demo_img.py
python3 scripts/01_demo_vid.py



# eval
python3 scripts/02_eval_lsotbtir.py



# eval chucnk
export CUDA_VISIBLE_DEVICES=0,1

python3 scripts/02_eval_lsotbtir_chunk.py \
--num_chunks 2 \
--chunk_idx 0 \
--device cuda:0 \
--exp_name sam2.1_TIR \
--save_to_video \
--model_name tiny

python3 scripts/02_eval_lsotbtir_chunk.py \
--num_chunks 2 \
--chunk_idx 1 \
--device cuda:1 \
--exp_name sam2.1_TIR \
--save_to_video \
--model_name tiny



python3 scripts/02_eval_lsotbtir_chunk.py \
--num_chunks 2 \
--chunk_idx 0 \
--device cuda:0 \
--exp_name sam2.1_TIR \
--save_to_video \
--model_name small

python3 scripts/02_eval_lsotbtir_chunk.py \
--num_chunks 2 \
--chunk_idx 1 \
--device cuda:1 \
--exp_name sam2.1_TIR \
--save_to_video \
--model_name small



python3 scripts/02_eval_lsotbtir_chunk.py \
--num_chunks 2 \
--chunk_idx 0 \
--device cuda:0 \
--exp_name sam2.1_TIR \
--save_to_video \
--model_name base_plus

python3 scripts/02_eval_lsotbtir_chunk.py \
--num_chunks 2 \
--chunk_idx 1 \
--device cuda:1 \
--exp_name sam2.1_TIR \
--save_to_video \
--model_name base_plus



python3 scripts/02_eval_lsotbtir_chunk.py \
--num_chunks 2 \
--chunk_idx 0 \
--device cuda:0 \
--exp_name sam2.1_TIR \
--save_to_video \
--model_name large

python3 scripts/02_eval_lsotbtir_chunk.py \
--num_chunks 2 \
--chunk_idx 1 \
--device cuda:1 \
--exp_name sam2.1_TIR \
--save_to_video \
--model_name large



# vis

python3 scripts/03_vis_lsotbtir.py



# Result


SAM 2.1 tiny


SAM 2.1 small

SAM 2.1 base_plus

SAM 2.1 large



# Detailed Result (per sequence)

