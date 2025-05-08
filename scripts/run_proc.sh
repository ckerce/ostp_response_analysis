for k in $(seq 9 10); do
  start="${k}876"
  end="${k}076"
  # Execute the python script directly
  python v2_analyze_rfi.py $(seq -s ' ' $start $end) > "v2_gemma3-27b-out_${start}.md"
done
