for i in {1..10}
do
  sbatch --output=/dev/null --error=/dev/null scripts/collect_fuzz_data.sh $i 0
done
