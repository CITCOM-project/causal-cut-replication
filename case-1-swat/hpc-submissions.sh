mkdir -p logs/attack_data
mkdir -p logs/attack_normal_data
mkdir -p logs/attack_mutated_normal_data
mkdir -p logs/attack_mutated_normal_data
for i in {0..30}
do
  sbatch hpc-run.sh successful_attacks_flat.json logs/attack_data/attack_$i.json $i ../data/attack_data_105.csv
  sbatch hpc-run.sh successful_attacks_flat.json logs/attack_normal_data/attack_$i.json $i ../data/attack_normal_data_105.csv
  sbatch hpc-run.sh successful_attacks_mutated.json logs/attack_mutated_attack_data/attack_$i.json $i ../data/attack_data_105.csv
  sbatch hpc-run.sh successful_attacks_mutated.json logs/attack_mutated_normal_data/attack_$i.json $i ../data/attack_normal_data_105.csv
done
