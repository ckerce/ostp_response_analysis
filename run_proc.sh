for k in $(seq 0 9); do
  start="${k}00"
  end="${k}99"
  # Execute the python script directly
  python v2_analyze_rfi.py $(seq -s ' ' $start $end) > "v2_gemma3-27b-out_${start}.md"
done
