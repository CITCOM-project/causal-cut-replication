mkdir -p logs/attack_data
mkdir -p logs/attack_normal_data
mkdir -p logs/attack_mutated_attack_data
mkdir -p logs/attack_mutated_normal_data
for i in {0..30}
do
  # This should run in a few hours
  python water_g.py -a successful_attacks_flat.json    -A -i $i -o logs/attack_data/attack_$i.json                ../data/attack_data_105.csv
  # This should run in a couple of days
  python water_g.py -a successful_attacks_flat.json    -A -i $i -o logs/attack_normal_data/attack_$i.json         ../data/attack_normal_data_105.csv
  # This should run in a few hours
  python water_g.py -a successful_attacks_mutated.json -A -i $i -o logs/attack_mutated_attack_data/attack_$i.json ../data/attack_data_105.csv
  # This should run in a couple of days
  python water_g.py -a successful_attacks_mutated.json -A -i $i -o logs/attack_mutated_normal_data/attack_$i.json ../data/attack_normal_data_105.csv
done
