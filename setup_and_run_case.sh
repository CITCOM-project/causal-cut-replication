case=$1

cd $case
python scripts/generate_configurations.py
# lines=$(wc -l < configurations.txt )
lines=1
cd ..

if command -v sbatch &>/dev/null; then
  # You can change the number passed to -P depending on how many CPU cores/how much memory you have
  seq 1 $lines | xargs -I {} -P 1 sbatch g_estimation.sh $case/configurations.txt {}
else
  seq 1 $lines | xargs -I {} bash g_estimation.sh $case/configurations.txt {}
fi
