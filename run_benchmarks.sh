#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:/home/users/kagyemang-dua/g5_adaptation:~/g5_adaptation

# List of games to run, manually specified to avoid parsing issues
GAMES="adventuregame hot_air_balloon codenames dond guesswhat imagegame matchit matchit_ascii mm_mapworld_graphs mm_mapworld mm_mapworld_specificroom multimodal_referencegame privateshared referencegame bbh cladder eqbench ifeval mmlu_pro taboo textmapworld_graphreasoning textmapworld textmapworld_specificroom privacy_qa wordle wordle_withclue wordle_withcritic"

for game in $GAMES; do
    echo "--- Starting game: $game ---"
    clem run -m Qwen3.5-9B-KTO -g "$game"
done
