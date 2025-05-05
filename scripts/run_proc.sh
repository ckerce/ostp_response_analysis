#for k in $(seq 0 9); do
#  start="${k}00"
#  end="${k}99"
#  # Execute the python script directly
#  python v2_analyze_rfi.py $(seq -s ' ' $start $end) > "v2_gemma3-27b-out_${start}.md"
#done

for k in $(seq 10062 -100 9062); do
  start="${k}"
  end="${k-100}"
  # Execute the python script directly
  python v2_analyze_rfi.py $(seq -s ' ' $start -1 $end) > "v2_gemma3-27b-out_${end}.md"
done
