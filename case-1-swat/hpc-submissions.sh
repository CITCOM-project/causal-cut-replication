mkdir -p logs/attack_data
mkdir -p logs/attack_normal_data
mkdir -p logs/attack_mutated_attack_data
mkdir -p logs/attack_mutated_normal_data
for i in {0..30}
do
  sbatch hpc-run.sh successful_attacks_flat.json    logs/attack_data_80/attack_$i.json                       0.20 $i ../data/attack_data_105.csv
  sbatch hpc-run.sh successful_attacks_flat.json    logs/attack_normal_data_80/attack_$i.json                0.20 $i ../data/attack_normal_data_105.csv
  sbatch hpc-run.sh successful_attacks_mutated.json logs/attack_mutated_attack_data_80/attack_$i.json        0.20 $i ../data/attack_data_105.csv
  sbatch hpc-run.sh successful_attacks_mutated.json logs/attack_mutated_attack_normal_data_80/attack_$i.json 0.20 $i ../data/attack_normal_data_105.csv

  sbatch hpc-run.sh successful_attacks_flat.json    logs/attack_data_90/attack_$i.json                0.10 $i ../data/attack_data_105.csv
  sbatch hpc-run.sh successful_attacks_flat.json    logs/attack_normal_data_90/attack_$i.json         0.10 $i ../data/attack_normal_data_105.csv
  sbatch hpc-run.sh successful_attacks_mutated.json logs/attack_mutated_attack_data_90/attack_$i.json 0.10 $i ../data/attack_data_105.csv
  sbatch hpc-run.sh successful_attacks_mutated.json logs/attack_mutated_attack_normal_data_90/attack_$i.json 0.10 $i ../data/attack_normal_data_105.csv

  sbatch hpc-run.sh successful_attacks_flat.json    logs/attack_data_95/attack_$i.json                0.05 $i ../data/attack_data_105.csv
  sbatch hpc-run.sh successful_attacks_flat.json    logs/attack_normal_data_95/attack_$i.json         0.05 $i ../data/attack_normal_data_105.csv
  sbatch hpc-run.sh successful_attacks_mutated.json logs/attack_mutated_attack_data_95/attack_$i.json 0.05 $i ../data/attack_data_105.csv
  sbatch hpc-run.sh successful_attacks_mutated.json logs/attack_mutated_attack_normal_data_95/attack_$i.json 0.05 $i ../data/attack_normal_data_105.csv
done
