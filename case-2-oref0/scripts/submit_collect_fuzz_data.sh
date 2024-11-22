for i in {1..69}
do
  sbatch --output=/dev/null --error=/dev/null scripts/collect_fuzz_data.sh $i 0
done
